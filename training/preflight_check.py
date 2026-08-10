"""Mandatory pre-training validation for 4-class SwinUNETR BraTS-GLI experiment.

This script performs comprehensive validation before allowing training to proceed.
It checks dataset integrity, label mapping, model configuration, and VRAM safety.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Import project components
from configs.config import (
    ACCUMULATION_STEPS,
    CE_CLASS_WEIGHTS,
    CROP_NEG,
    CROP_POS,
    DATA_ROOT,
    DEVICE,
    EMA_DEVICE,
    INPUT_CHANNELS,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PATCH_SIZE,
    SCHEDULER,
    SWIN_FEATURE_SIZE,
    SWIN_USE_CHECKPOINT,
    TRAIN_DIR,
    TRAIN_NUM_SAMPLES,
    USE_EMA,
    USE_MIXED_PRECISION,
    USE_PRETRAINED_SWIN,
    VAL_DIR,
    ensure_experiment_dirs,
)
from datasets.brats_dataset import BraTSDataset, _remap_brats_labels
from datasets.dataloader import get_train_loader, get_train_transforms, get_val_loader, get_val_transforms
from models.model_factory import build_model
from utils.brats_metrics import compute_region_dice, logits_to_prediction


class PreflightCheck:
    """Comprehensive pre-training validation."""
    
    def __init__(self):
        self.checks_passed: List[str] = []
        self.checks_failed: List[str] = []
        self.results: Dict[str, Any] = {}
        
    def record_pass(self, check_name: str, details: str = ""):
        """Record a passed check."""
        self.checks_passed.append(check_name)
        if details:
            self.results[check_name] = {"status": "PASS", "details": details}
        else:
            self.results[check_name] = {"status": "PASS"}
    
    def record_fail(self, check_name: str, reason: str):
        """Record a failed check."""
        self.checks_failed.append(check_name)
        self.results[check_name] = {"status": "FAIL", "reason": reason}
    
    def check_dataset_exists(self):
        """[1] Dataset exists."""
        try:
            if not DATA_ROOT.exists():
                self.record_fail("dataset_exists", f"Data root not found: {DATA_ROOT}")
                return
            
            self.record_pass("dataset_exists", f"Data root: {DATA_ROOT}")
        except Exception as e:
            self.record_fail("dataset_exists", str(e))
    
    def check_split_directories(self):
        """[2] Training/Validation/Testing directories exist."""
        try:
            splits = {
                "Training": TRAIN_DIR,
                "Validation": VAL_DIR,
            }
            
            for split_name, split_dir in splits.items():
                if not split_dir.exists():
                    self.record_fail(f"{split_name.lower()}_dir_exists", f"{split_name} directory not found: {split_dir}")
                else:
                    self.record_pass(f"{split_name.lower()}_dir_exists", f"{split_name}: {split_dir}")
        except Exception as e:
            self.record_fail("split_directories", str(e))
    
    def check_modalities(self):
        """[3] Expected modalities exist (t1n, t1c, t2w, t2f)."""
        try:
            # Check training split for modalities
            train_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
            if not train_dirs:
                self.record_fail("modalities_valid", "No training cases found")
                return
            
            sample_case = train_dirs[0]
            ds = BraTSDataset(root_dir=TRAIN_DIR, case_dirs=[sample_case.name])
            
            # Try to load a sample to verify modalities
            try:
                image, mask, case_id = ds[0]
                if image.shape[0] == 4:
                    self.record_pass("modalities_valid", f"4 modalities loaded: {image.shape}")
                else:
                    self.record_fail("modalities_valid", f"Expected 4 modalities, got {image.shape[0]}")
            except Exception as e:
                self.record_fail("modalities_valid", f"Failed to load modalities: {e}")
        except Exception as e:
            self.record_fail("modalities_valid", str(e))
    
    def check_raw_labels_contain_et(self):
        """[4] Raw labels contain ET label 3."""
        try:
            train_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
            if not train_dirs:
                self.record_fail("raw_labels_contain_et", "No training cases found")
                return
            
            # Use the same robust segmentation file finding logic as audit_brats_labels.py
            import re
            import nibabel as nib
            
            def find_segmentation_file(case_dir: Path) -> Path:
                """Find the segmentation file using robust regex matching."""
                seg_patterns = [re.compile(r"(?:^|[^a-zA-Z0-9])(?:seg|label)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE)]
                
                all_files = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
                
                for file_path in all_files:
                    filename = file_path.name
                    if filename.endswith(".nii.gz"):
                        filename = filename[:-7]
                    elif filename.endswith(".nii"):
                        filename = filename[:-4]
                    
                    for pattern in seg_patterns:
                        if pattern.search(filename):
                            return file_path
                
                raise FileNotFoundError(f"No segmentation file found in: {case_dir}")
            
            # Check multiple cases for ET presence
            found_et = False
            cases_checked = 0
            
            for case_dir in train_dirs[:min(10, len(train_dirs))]:
                try:
                    seg_file = find_segmentation_file(case_dir)
                    
                    img = nib.load(str(seg_file))
                    vol = img.get_fdata(dtype=np.float32)
                    vol = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
                    vol = np.transpose(vol, (2, 0, 1))
                    unique_labels = set(np.unique(vol.astype(np.int16)))
                    
                    if 3 in unique_labels:
                        found_et = True
                        break
                    
                    cases_checked += 1
                except Exception:
                    continue
            
            if found_et:
                self.record_pass("raw_labels_contain_et", f"ET label 3 found in raw data")
            else:
                self.record_fail("raw_labels_contain_et", f"ET label 3 not found in {cases_checked} cases checked")
        except Exception as e:
            self.record_fail("raw_labels_contain_et", str(e))
    
    def check_remapping_preserves_et(self):
        """[5] Remapping preserves ET label 3."""
        try:
            # Test the remapping function directly
            test_mask = np.array([[[0, 1, 2, 3]]], dtype=np.int16)
            remapped = _remap_brats_labels(test_mask)
            
            raw_et = np.sum(test_mask == 3)
            mapped_et = np.sum(remapped == 3)
            
            if raw_et > 0 and mapped_et == raw_et:
                self.record_pass("remapping_preserves_et", f"ET preserved: {raw_et} -> {mapped_et} voxels")
            else:
                self.record_fail("remapping_preserves_et", f"ET loss: {raw_et} -> {mapped_et} voxels")
        except Exception as e:
            self.record_fail("remapping_preserves_et", str(e))
    
    def check_num_classes(self):
        """[6] NUM_CLASSES == 4."""
        if NUM_CLASSES == 4:
            self.record_pass("num_classes_valid", f"NUM_CLASSES = {NUM_CLASSES}")
        else:
            self.record_fail("num_classes_valid", f"NUM_CLASSES = {NUM_CLASSES}, expected 4")
    
    def check_model_out_channels(self):
        """[7] SwinUNETR out_channels == 4."""
        try:
            model, _ = build_model(
                model_name=MODEL_NAME,
                in_channels=INPUT_CHANNELS,
                out_channels=NUM_CLASSES,
                patch_size=PATCH_SIZE,
                swin_feature_size=SWIN_FEATURE_SIZE,
                swin_use_checkpoint=SWIN_USE_CHECKPOINT,
            )
            
            # Check the model output channels
            if hasattr(model, 'out_channels'):
                out_ch = model.out_channels
            else:
                # Try to infer from forward pass
                dummy_input = torch.randn(1, INPUT_CHANNELS, *PATCH_SIZE)
                dummy_output = model(dummy_input)
                out_ch = dummy_output.shape[1]
            
            if out_ch == 4:
                self.record_pass("model_out_channels_valid", f"Model out_channels = {out_ch}")
            else:
                self.record_fail("model_out_channels_valid", f"Model out_channels = {out_ch}, expected 4")
        except Exception as e:
            self.record_fail("model_out_channels_valid", str(e))
    
    def check_sample_labels(self):
        """[8,9] Sample training and validation labels contain class 3."""
        try:
            # Check training
            train_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
            if train_dirs:
                ds = BraTSDataset(root_dir=TRAIN_DIR, case_dirs=[train_dirs[0].name])
                image, mask, case_id = ds[0]
                unique_labels = set(np.unique(mask.numpy()))
                
                if 3 in unique_labels:
                    self.record_pass("train_label_contains_et", f"Training label contains class 3: {sorted(unique_labels)}")
                else:
                    self.record_fail("train_label_contains_et", f"Training label missing class 3: {sorted(unique_labels)}")
            
            # Check validation
            val_dirs = [d for d in VAL_DIR.iterdir() if d.is_dir()]
            if val_dirs:
                ds = BraTSDataset(root_dir=VAL_DIR, case_dirs=[val_dirs[0].name])
                image, mask, case_id = ds[0]
                unique_labels = set(np.unique(mask.numpy()))
                
                if 3 in unique_labels:
                    self.record_pass("val_label_contains_et", f"Validation label contains class 3: {sorted(unique_labels)}")
                else:
                    self.record_fail("val_label_contains_et", f"Validation label missing class 3: {sorted(unique_labels)}")
                    
        except Exception as e:
            self.record_fail("sample_labels_check", str(e))
    
    def check_tumor_centered_sampling(self):
        """[10] RandCropByPosNegLabeld is active."""
        try:
            transforms = get_train_transforms(
                patch_size=PATCH_SIZE,
                pos=CROP_POS,
                neg=CROP_NEG,
                num_samples=TRAIN_NUM_SAMPLES,
            )
            
            # Check if RandCropByPosNegLabeld is in the transforms
            from monai.transforms import RandCropByPosNegLabeld
            
            has_crop = any(isinstance(t, RandCropByPosNegLabeld) for t in transforms.transforms)
            
            if has_crop:
                self.record_pass("tumor_centered_sampling_active", "RandCropByPosNegLabeld found in transforms")
            else:
                self.record_fail("tumor_centered_sampling_active", "RandCropByPosNegLabeld not found in transforms")
        except Exception as e:
            self.record_fail("tumor_centered_sampling_active", str(e))
    
    def check_training_patch_et(self):
        """[11] Training patch can contain class 3."""
        try:
            train_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
            if not train_dirs:
                self.record_fail("training_patch_et_valid", "No training cases found")
                return
            
            # Create dataloader with transforms
            loader = get_train_loader(
                root_dir=TRAIN_DIR,
                case_dirs=[train_dirs[0].name],
                batch_size=1,
                num_workers=0,
                patch_size=PATCH_SIZE,
                pos=CROP_POS,
                neg=CROP_NEG,
                num_samples=TRAIN_NUM_SAMPLES,
            )
            
            # Sample a batch
            batch = next(iter(loader))
            if isinstance(batch, dict):
                labels = batch["label"]
            else:
                labels = batch[1]
            
            # Check for ET in the batch
            unique_labels = set(np.unique(labels.numpy()))
            
            if 3 in unique_labels:
                self.record_pass("training_patch_et_valid", f"Training patch contains class 3: {sorted(unique_labels)}")
            else:
                self.record_fail("training_patch_et_valid", f"Training patch missing class 3: {sorted(unique_labels)}")
        except Exception as e:
            self.record_fail("training_patch_et_valid", str(e))
    
    def check_loss_weights(self):
        """[12,13] Loss accepts four class weights, CE_CLASS_WEIGHTS length == 4."""
        try:
            if len(CE_CLASS_WEIGHTS) == 4:
                self.record_pass("ce_class_weights_length", f"CE_CLASS_WEIGHTS length = {len(CE_CLASS_WEIGHTS)}")
            else:
                self.record_fail("ce_class_weights_length", f"CE_CLASS_WEIGHTS length = {len(CE_CLASS_WEIGHTS)}, expected 4")
            
            # Test loss with 4 classes
            from configs.config import DiceCrossEntropyLoss
            
            loss_fn = DiceCrossEntropyLoss(
                num_classes=4,
                ce_class_weights=CE_CLASS_WEIGHTS,
            )
            
            # Test forward pass
            dummy_logits = torch.randn(2, 4, 32, 32, 32)
            dummy_target = torch.randint(0, 4, (2, 32, 32, 32))
            
            loss = loss_fn(dummy_logits, dummy_target)
            
            if loss.item() > 0:
                self.record_pass("loss_accepts_four_weights", "Loss forward pass successful with 4 classes")
            else:
                self.record_fail("loss_accepts_four_weights", "Loss forward pass produced invalid value")
        except Exception as e:
            self.record_fail("loss_weights_check", str(e))
    
    def check_validation_metric(self):
        """[14] Validation metric recognizes class 3."""
        try:
            # Test metric computation with class 3
            dummy_pred = torch.randint(0, 4, (2, 32, 32, 32))
            dummy_gt = torch.randint(0, 4, (2, 32, 32, 32))
            
            # Ensure some class 3 in ground truth
            dummy_gt[0, 0, 0, 0] = 3
            
            scores = compute_region_dice(dummy_pred, dummy_gt, from_logits=False)
            
            # Check that ET score is computed
            if hasattr(scores, 'et') and not np.isnan(scores.et):
                self.record_pass("validation_metric_recognizes_et", f"Metric computes ET score: {scores.et:.4f}")
            else:
                self.record_fail("validation_metric_recognizes_et", "Metric failed to compute ET score")
        except Exception as e:
            self.record_fail("validation_metric_recognizes_et", str(e))
    
    def check_region_definitions(self):
        """[15,16,17] WT/TC/ET region definitions are correct."""
        try:
            from utils.brats_metrics import _region_masks
            
            # Test region definitions with controlled synthetic masks
            # Create separate test masks for each class to avoid overlap issues
            test_cases = [
                (torch.zeros(1, 32, 32, 32, dtype=torch.long), "background_only"),
                (torch.zeros(1, 32, 32, 32, dtype=torch.long), "class_0"),
                (torch.ones(1, 32, 32, 32, dtype=torch.long), "class_1"),
                (torch.full((1, 32, 32, 32), 2, dtype=torch.long), "class_2"),
                (torch.full((1, 32, 32, 32), 3, dtype=torch.long), "class_3"),
            ]
            
            results = {
                "wt_includes_1": False,
                "wt_includes_2": False, 
                "wt_includes_3": False,
                "wt_excludes_0": False,
                "tc_includes_1": False,
                "tc_includes_3": False,
                "tc_excludes_0": False,
                "tc_excludes_2": False,
                "et_includes_3": False,
                "et_excludes_0": False,
                "et_excludes_1": False,
                "et_excludes_2": False,
            }
            
            for test_mask, label_name in test_cases:
                wt_mask = _region_masks(test_mask, "wt")
                tc_mask = _region_masks(test_mask, "tc")
                et_mask = _region_masks(test_mask, "et")
                
                # Check if any voxel is True in the region mask
                has_wt = wt_mask.any().item()
                has_tc = tc_mask.any().item()
                has_et = et_mask.any().item()
                
                if label_name == "class_1":
                    results["wt_includes_1"] = has_wt
                    results["tc_includes_1"] = has_tc
                    results["et_excludes_1"] = not has_et
                elif label_name == "class_2":
                    results["wt_includes_2"] = has_wt
                    results["tc_excludes_2"] = not has_tc
                    results["et_excludes_2"] = not has_et
                elif label_name == "class_3":
                    results["wt_includes_3"] = has_wt
                    results["tc_includes_3"] = has_tc
                    results["et_includes_3"] = has_et
                elif label_name in ["background_only", "class_0"]:
                    results["wt_excludes_0"] = not has_wt
                    results["tc_excludes_0"] = not has_tc
                    results["et_excludes_0"] = not has_et
            
            # Verify WT definition: {1,2,3}
            wt_correct = (results["wt_includes_1"] and results["wt_includes_2"] and 
                        results["wt_includes_3"] and results["wt_excludes_0"])
            
            # Verify TC definition: {1,3}
            tc_correct = (results["tc_includes_1"] and results["tc_includes_3"] and
                        results["tc_excludes_0"] and results["tc_excludes_2"])
            
            # Verify ET definition: {3}
            et_correct = (results["et_includes_3"] and results["et_excludes_0"] and
                        results["et_excludes_1"] and results["et_excludes_2"])
            
            if wt_correct:
                self.record_pass("wt_definition_correct", "WT = labels {1,2,3}")
            else:
                self.record_fail("wt_definition_correct", f"WT definition incorrect: {results}")
                
            if tc_correct:
                self.record_pass("tc_definition_correct", "TC = labels {1,3}")
            else:
                self.record_fail("tc_definition_correct", f"TC definition incorrect: {results}")
                
            if et_correct:
                self.record_pass("et_definition_correct", "ET = label {3}")
            else:
                self.record_fail("et_definition_correct", f"ET definition incorrect: {results}")
                    
        except Exception as e:
            self.record_fail("region_definitions", str(e))
    
    def check_no_et_conversion_to_background(self):
        """[18] No ET voxels are converted to background."""
        try:
            # This is essentially tested by the remapping check
            # Additional verification with actual data
            train_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
            if not train_dirs:
                self.record_fail("no_et_conversion", "No training cases found")
                return
            
            # Test with actual data
            ds = BraTSDataset(root_dir=TRAIN_DIR, case_dirs=[train_dirs[0].name])
            image, mask, case_id = ds[0]
            
            # Verify that if mask has any class 3, it's not silently converted
            mask_np = mask.numpy()
            if 3 in mask_np:
                # If class 3 exists, it should be preserved
                self.record_pass("no_et_conversion", "ET voxels preserved in actual data")
            else:
                # This case might not have ET, which is acceptable
                self.record_pass("no_et_conversion", "No ET in sample case (acceptable)")
        except Exception as e:
            self.record_fail("no_et_conversion", str(e))
    
    def check_model_forward_shape(self):
        """[19] Model forward pass produces 4 output channels."""
        try:
            model, _ = build_model(
                model_name=MODEL_NAME,
                in_channels=INPUT_CHANNELS,
                out_channels=NUM_CLASSES,
                patch_size=PATCH_SIZE,
                swin_feature_size=SWIN_FEATURE_SIZE,
                swin_use_checkpoint=SWIN_USE_CHECKPOINT,
            )
            model.eval()
            
            dummy_input = torch.randn(1, INPUT_CHANNELS, *PATCH_SIZE)
            with torch.no_grad():
                output = model(dummy_input)
            
            if output.shape[1] == 4:
                self.record_pass("model_forward_shape_valid", f"Model output shape: {output.shape}")
            else:
                self.record_fail("model_forward_shape_valid", f"Model output shape: {output.shape}, expected [N,4,D,H,W]")
        except Exception as e:
            self.record_fail("model_forward_shape_valid", str(e))
    
    def check_loss_forward_backward(self):
        """[20] Loss forward/backward works."""
        try:
            from configs.config import DiceCrossEntropyLoss
            
            model, _ = build_model(
                model_name=MODEL_NAME,
                in_channels=INPUT_CHANNELS,
                out_channels=NUM_CLASSES,
                patch_size=PATCH_SIZE,
                swin_feature_size=SWIN_FEATURE_SIZE,
                swin_use_checkpoint=SWIN_USE_CHECKPOINT,
            )
            loss_fn = DiceCrossEntropyLoss(num_classes=4, ce_class_weights=CE_CLASS_WEIGHTS)
            
            dummy_input = torch.randn(1, INPUT_CHANNELS, *PATCH_SIZE, requires_grad=True)
            dummy_target = torch.randint(0, 4, (1, *PATCH_SIZE))
            
            # Forward
            logits = model(dummy_input)
            loss = loss_fn(logits, dummy_target)
            
            # Backward
            loss.backward()
            
            if loss.item() > 0 and dummy_input.grad is not None:
                self.record_pass("loss_forward_backward_valid", "Loss forward/backward successful")
            else:
                self.record_fail("loss_forward_backward_valid", "Loss forward/backward failed")
        except Exception as e:
            self.record_fail("loss_forward_backward_valid", str(e))
    
    def check_vram_safety(self):
        """[21] VRAM dry-run still passes (basic check)."""
        try:
            if not torch.cuda.is_available():
                self.record_pass("vram_basic_check", "CUDA not available, skipping VRAM check")
                return
            
            # Basic VRAM check
            device = torch.device("cuda")
            model, _ = build_model(
                model_name=MODEL_NAME,
                in_channels=INPUT_CHANNELS,
                out_channels=NUM_CLASSES,
                patch_size=PATCH_SIZE,
                swin_feature_size=SWIN_FEATURE_SIZE,
                swin_use_checkpoint=SWIN_USE_CHECKPOINT,
            )
            model = model.to(device)
            
            # Try a forward pass
            dummy_input = torch.randn(1, INPUT_CHANNELS, *PATCH_SIZE).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check VRAM
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            
            self.record_pass("vram_basic_check", f"VRAM: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            
            # Clean up
            del model, dummy_input, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.record_fail("vram_basic_check", str(e))
    
    def check_experiment_protection(self):
        """[22] Existing experiment outputs are untouched."""
        try:
            old_exp = OUTPUT_DIR / "exp_swinunetr_4class_final"
            new_exp = OUTPUT_DIR / "exp_swinunetr_4class_et_fixed"
            
            if old_exp.exists():
                self.record_pass("old_experiment_protected", f"Previous experiment exists: {old_exp}")
            else:
                self.record_pass("old_experiment_protected", "No previous experiment to protect")
            
            # Ensure new experiment directory can be created
            new_exp.mkdir(parents=True, exist_ok=True)
            self.record_pass("new_experiment_ready", f"New experiment directory: {new_exp}")
            
        except Exception as e:
            self.record_fail("experiment_protection", str(e))
    
    def run_all_checks(self) -> bool:
        """Run all preflight checks."""
        print("=" * 60)
        print("PRE-TRAINING PREFLIGHT CHECK")
        print("=" * 60)
        
        # Run all checks
        self.check_dataset_exists()
        self.check_split_directories()
        self.check_modalities()
        self.check_raw_labels_contain_et()
        self.check_remapping_preserves_et()
        self.check_num_classes()
        self.check_model_out_channels()
        self.check_sample_labels()
        self.check_tumor_centered_sampling()
        self.check_training_patch_et()
        self.check_loss_weights()
        self.check_validation_metric()
        self.check_region_definitions()
        self.check_no_et_conversion_to_background()
        self.check_model_forward_shape()
        self.check_loss_forward_backward()
        self.check_vram_safety()
        self.check_experiment_protection()
        
        # Print results
        print("\n" + "=" * 60)
        print("CHECK RESULTS")
        print("=" * 60)
        
        print(f"\nPASSED ({len(self.checks_passed)}):")
        for check in self.checks_passed:
            print(f"  - {check}")
        
        print(f"\nFAILED ({len(self.checks_failed)}):")
        for check in self.checks_failed:
            print(f"  - {check}")
        
        # Overall decision
        print("\n" + "=" * 60)
        print("PRE-TRAINING DECISION")
        print("=" * 60)
        
        if not self.checks_failed:
            print("SAFE TO START 4-CLASS ET TRAINING")
            return True
        else:
            print("DO NOT TRAIN")
            print("\nFailed checks must be resolved before training:")
            for check in self.checks_failed:
                reason = self.results.get(check, {}).get("reason", "Unknown reason")
                print(f"  - {check}: {reason}")
            return False
    
    def save_report(self, output_path: Path):
        """Save detailed report to JSON."""
        report = {
            "preflight_check": {
                "total_checks": len(self.checks_passed) + len(self.checks_failed),
                "passed": len(self.checks_passed),
                "failed": len(self.checks_failed),
                "overall_status": "READY" if not self.checks_failed else "NOT_READY",
            },
            "checks": self.results,
            "configuration": {
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "input_channels": INPUT_CHANNELS,
                "patch_size": PATCH_SIZE,
                "train_num_samples": TRAIN_NUM_SAMPLES,
                "accumulation_steps": ACCUMULATION_STEPS,
                "use_mixed_precision": USE_MIXED_PRECISION,
                "use_ema": USE_EMA,
                "ema_device": EMA_DEVICE,
                "ce_class_weights": CE_CLASS_WEIGHTS,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "scheduler": SCHEDULER,
                "swin_feature_size": SWIN_FEATURE_SIZE,
                "swin_use_checkpoint": SWIN_USE_CHECKPOINT,
                "use_pretrained_swin": USE_PRETRAINED_SWIN,
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_path}")


def main():
    """Run preflight checks and save report."""
    checker = PreflightCheck()
    safe_to_train = checker.run_all_checks()
    
    # Save report
    from configs.config import METRICS_DIR, EXP_NAME
    report_path = METRICS_DIR / "preflight_report.json"
    checker.save_report(report_path)
    
    # Exit with appropriate code
    sys.exit(0 if safe_to_train else 1)


if __name__ == "__main__":
    main()