"""READ-ONLY audit tool for BraTS-GLI dataset labels.

This script scans the dataset and reports label statistics without modifying any files.
It validates that the BraTS-GLI label convention {0,1,2,3} is correctly preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np


def load_nifti_labels(path: Path) -> np.ndarray:
    """Load NIfTI segmentation mask and return as numpy array."""
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    vol = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
    vol = np.transpose(vol, (2, 0, 1))
    return vol.astype(np.int16)


def find_segmentation_file(case_dir: Path) -> Path:
    """Find the segmentation file in a case directory."""
    import re

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


def audit_split(split_dir: Path, split_name: str) -> Dict:
    """Audit a single dataset split (Training/Validation/Testing)."""
    if not split_dir.exists():
        print(f"WARNING: {split_name} directory not found: {split_dir}")
        return None
    
    case_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    
    if not case_dirs:
        print(f"WARNING: No case directories found in {split_name}: {split_dir}")
        return None
    
    print(f"\nDATASET LABEL AUDIT")
    print(f"{'-' * 40}")
    print(f"Split: {split_name}")
    print(f"Cases: {len(case_dirs)}")
    
    all_raw_labels = set()
    all_mapped_labels = set()
    total_et_voxels = 0
    cases_with_et = 0
    cases_without_et = 0
    et_voxel_counts = []
    
    for case_dir in case_dirs:
        try:
            seg_file = find_segmentation_file(case_dir)
            raw_mask = load_nifti_labels(seg_file)
            
            # Track raw labels
            raw_unique = set(np.unique(raw_mask))
            all_raw_labels.update(raw_unique)
            
            # Apply the same remapping logic as the dataset
            mask_i = raw_mask.astype(np.int16, copy=False)
            mapped_mask = np.zeros_like(mask_i, dtype=np.int16)
            mapped_mask[mask_i == 1] = 1
            mapped_mask[mask_i == 2] = 2
            mapped_mask[mask_i == 3] = 3
            
            # Track mapped labels
            mapped_unique = set(np.unique(mapped_mask))
            all_mapped_labels.update(mapped_unique)
            
            # ET statistics
            raw_et_voxels = np.sum(raw_mask == 3)
            mapped_et_voxels = np.sum(mapped_mask == 3)
            
            if raw_et_voxels > 0:
                cases_with_et += 1
                et_voxel_counts.append(raw_et_voxels)
                total_et_voxels += raw_et_voxels
                
                # Validate ET preservation
                if mapped_et_voxels != raw_et_voxels:
                    print(f"WARNING: ET loss in {case_dir.name}: raw={raw_et_voxels}, mapped={mapped_et_voxels}")
            else:
                cases_without_et += 1
                
        except Exception as e:
            print(f"WARNING: Error processing {case_dir.name}: {e}")
            continue
    
    print(f"Raw labels: {sorted(all_raw_labels)}")
    print(f"Mapped labels: {sorted(all_mapped_labels)}")
    print(f"Cases with ET: {cases_with_et}/{len(case_dirs)}")
    print(f"Cases without ET: {cases_without_et}/{len(case_dirs)}")
    print(f"Total ET voxels: {total_et_voxels}")
    
    if et_voxel_counts:
        print(f"Min ET voxels: {min(et_voxel_counts)}")
        print(f"Max ET voxels: {max(et_voxel_counts)}")
        print(f"Mean ET voxels: {int(np.mean(et_voxel_counts))}")
    else:
        print(f"Min ET voxels: 0")
        print(f"Max ET voxels: 0")
        print(f"Mean ET voxels: 0")
    
    return {
        "split": split_name,
        "cases": len(case_dirs),
        "raw_labels": sorted(all_raw_labels),
        "mapped_labels": sorted(all_mapped_labels),
        "cases_with_et": cases_with_et,
        "cases_without_et": cases_without_et,
        "total_et_voxels": int(total_et_voxels),
        "min_et_voxels": min(et_voxel_counts) if et_voxel_counts else 0,
        "max_et_voxels": max(et_voxel_counts) if et_voxel_counts else 0,
        "mean_et_voxels": int(np.mean(et_voxel_counts)) if et_voxel_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit BraTS-GLI dataset labels")
    parser.add_argument("--data-root", type=str, default=None, help="Path to BraTS data root")
    args = parser.parse_args()
    
    # Determine data root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        # Use the same path calculation as configs/config.py
        # Path(__file__).resolve().parents[1] gives brain-tumor-segmentation
        # Then parent gives BrainTumorSegmentation, then add "BraTS"
        project_root = Path(__file__).resolve().parents[2]  # Goes up to BrainTumorSegmentation
        data_root = project_root / "BraTS"
    
    print(f"Auditing dataset at: {data_root}")
    
    # Audit each split
    splits = {
        "Training": data_root / "Training",
        "Validation": data_root / "Validation", 
        "Testing": data_root / "Testing",
    }
    
    results = {}
    for split_name, split_dir in splits.items():
        result = audit_split(split_dir, split_name)
        if result:
            results[split_name] = result
    
    # Summary
    print(f"\n{'=' * 40}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 40}")
    
    if results:
        for split_name, result in results.items():
            print(f"{split_name}: {result['cases']} cases, ET in {result['cases_with_et']} cases")
        
        # Validate BraTS-GLI convention
        all_raw = set()
        all_mapped = set()
        for result in results.values():
            all_raw.update(result["raw_labels"])
            all_mapped.update(result["mapped_labels"])
        
        print(f"\nOverall raw labels: {sorted(all_raw)}")
        print(f"Overall mapped labels: {sorted(all_mapped)}")
        
        expected_brats_gli = {0, 1, 2, 3}
        if all_raw == expected_brats_gli and all_mapped == expected_brats_gli:
            print("Dataset follows BraTS-GLI convention {0,1,2,3}")
        elif all_raw == {0, 1, 2, 4} and all_mapped == {0, 1, 2, 3}:
            print("WARNING: Dataset follows legacy BraTS convention {0,1,2,4} → {0,1,2,3}")
        else:
            print(f"ERROR: Unexpected label convention detected!")
            print(f"   Expected BraTS-GLI: {sorted(expected_brats_gli)}")
            print(f"   Found raw: {sorted(all_raw)}")
            print(f"   Found mapped: {sorted(all_mapped)}")
    else:
        print("ERROR: No valid splits found to audit")


if __name__ == "__main__":
    main()