# Product Requirements Document (PRD)
# AI-Assisted Brain Tumor Segmentation

**Project:** BrainTumorSegmentation
**Author:** Final Year Computer Science Project
**Document Version:** 1.0
**Last Updated:** 2026-08-04
**Status:** Implementation Phase — Data Pipeline Complete, Training in Progress

---

## 1. Document Control

| Field | Value |
|---|---|
| Project Codename | KP-BrainTumorSegmentation |
| Project Type | Final Year Capstone — Deep Learning / Medical Imaging |
| Target Domain | Clinical Decision Support (Neuro-oncology) |
| Repository Path | `C:\Users\loq\OneDrive\Desktop\KP\BrainTumorSegmentation` |
| Primary Codebase | `brain-tumor-segmentation/` |
| Dataset | BraTS (Brain Tumor Segmentation) Challenge |
| Primary Stack | Python, PyTorch, MONAI, NiBabel, NumPy |

---

## 2. Executive Summary

This project delivers an **AI-assisted 3D brain tumor segmentation system** that automatically identifies and delineates tumor sub-regions from multi-modal MRI scans (T1, T1ce, T2, FLAIR). The system is intended to **assist radiologists and clinicians** by reducing the manual effort required to annotate volumetric MRI data, accelerating diagnosis, and supporting objective, reproducible treatment planning.

The product is a research-grade deep learning pipeline built around a **3D U-Net architecture** (with optional SwinUNETR transformer variant) trained on the publicly available BraTS dataset. The system supports both **patch-based training** (for memory-constrained GPUs such as the RTX 4050 6GB) and **sliding-window inference** for full-volume predictions, producing clinically interpretable outputs (segmentation masks, probability maps, and qualitative visualizations).

---

## 3. Problem Statement

### 3.1 Background
Gliomas are the most common primary brain tumors in adults. Accurate segmentation of tumor sub-regions — **Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)** — is critical for:
- Diagnosis and grading
- Surgical planning
- Radiation therapy targeting
- Longitudinal treatment response monitoring

### 3.2 Current Pain Points
Manual segmentation of a single 3D MRI volume takes a trained radiologist **15–60 minutes** and is subject to:
- **Inter-observer variability** (different clinicians produce different boundaries)
- **Intra-observer variability** (same clinician differs across sessions)
- **Scalability bottlenecks** in high-volume clinical settings
- **Fatigue and error** over long annotation sessions

### 3.3 Opportunity
Deep learning models, particularly 3D U-Net variants, have demonstrated Dice scores above 0.85 on BraTS benchmarks. A well-engineered, reproducible pipeline can deliver near-expert segmentation quality in seconds per case.

---

## 4. Goals & Non-Goals

### 4.1 Goals (In-Scope)
1. **Build a production-quality 3D segmentation pipeline** for BraTS-format MRI data.
2. **Achieve competitive segmentation accuracy** measured by class-averaged Dice Score on the BraTS validation set.
3. **Support multiple model architectures** (Baseline 3D U-Net, Residual Attention 3D U-Net, MONAI SwinUNETR) selectable via configuration.
4. **Run efficiently on consumer-grade hardware** (6GB VRAM GPU like RTX 4050) using patch-based training, mixed precision, and gradient checkpointing.
5. **Provide a complete MLOps-style workflow**: configuration, training, validation, checkpointing, experiment logging, inference, and visualization.
6. **Produce qualitative outputs** (2D slice overlays, 3D meshes) for clinical interpretability and academic presentation.

### 4.2 Non-Goals (Out-of-Scope)
1. **Clinical deployment / FDA / CE marking** — this is a research/academic project, not a regulated medical device.
2. **Real-time inference** — inference time is not latency-critical for offline analysis.
3. **Multi-disease support** — scope is limited to gliomas via the BraTS dataset.
4. **DICOM I/O** — input is NIfTI (`.nii` / `.nii.gz`); DICOM conversion is out of scope.
5. **Web / mobile UI** — CLI-based inference only.
6. **Tumor sub-typing (IDH mutation, 1p/19q codeletion)** — purely a segmentation task.

---

## 5. Target Users & Use Cases

### 5.1 Primary Users
- **Radiologists / Neuroradiologists** — second-reader tool to accelerate annotation.
- **Radiation oncologists** — target volume definition for treatment planning.
- **Neurosurgeons** — pre-surgical tumor mapping.

### 5.2 Secondary Users
- **Medical imaging researchers** — reproducible baseline for novel architectures.
- **ML engineers** — reference implementation for 3D medical segmentation.

### 5.3 Use Cases
| ID | Use Case | Actor | Outcome |
|---|---|---|---|
| UC-1 | Segment a new MRI volume | Clinician | Tumor sub-regions labeled in < 30 seconds |
| UC-2 | Batch-process a study cohort | Researcher | Consistent, reproducible segmentations across all cases |
| UC-3 | Compare model architectures | ML Engineer | Train & evaluate baseline vs residual vs SwinUNETR |
| UC-4 | Visualize predictions for review | Clinician | 2D overlays + 3D meshes exported for presentation |
| UC-5 | Resume interrupted training | Researcher | Checkpoint-based restart from last epoch |

---

## 6. Success Metrics

### 6.1 Primary Metric
- **Mean Dice Score (across 3 tumor classes: WT/TC/ET)** on the BraTS validation set.

### 6.2 Secondary Metrics
- **Per-class Dice** for Whole Tumor, Tumor Core, Enhancing Tumor.
- **Hausdorff Distance (95th percentile)** for boundary accuracy.
- **Training stability** — convergence within configured epoch budget (75 epochs default).
- **Inference latency** — full-volume inference time per case.
- **Memory footprint** — peak GPU VRAM during training (target: ≤ 6 GB).

### 6.3 Success Criteria
| Tier | Mean Dice | Interpretation |
|---|---|---|
| Minimum Viable | ≥ 0.70 | Reproducible baseline; demonstrates end-to-end pipeline |
| Target | ≥ 0.82 | Competitive with published BraTS baselines |
| Stretch | ≥ 0.88 | Near-expert performance |

### 6.4 Engineering Success
- Pipeline trains end-to-end without errors on a single RTX 4050 (6 GB).
- All three model variants train successfully with the same data pipeline.
- Checkpointing, experiment logging, and inference scripts run without manual intervention.
- Documentation (this PRD, README) sufficient for a new contributor to reproduce results.

---

## 7. Functional Requirements

### 7.1 Data Ingestion
| ID | Requirement | Priority |
|---|---|---|
| FR-D1 | Load multi-modal MRI volumes (T1, T1ce, T2, FLAIR) in NIfTI format | P0 |
| FR-D2 | Load corresponding segmentation mask (`.nii.gz`) | P0 |
| FR-D3 | Convert to canonical orientation via `nib.as_closest_canonical` | P0 |
| FR-D4 | Transpose to `[D, H, W]` orientation | P0 |
| FR-D5 | Remap BraTS labels {0,1,2,4} → {0,1,2,3} for contiguous class indices | P0 |
| FR-D6 | Apply per-volume z-score normalization | P0 |
| FR-D7 | Pad volumes smaller than patch size to prevent crashes | P0 |

### 7.2 Data Pipeline
| ID | Requirement | Priority |
|---|---|---|
| FR-P1 | Extract random 96×96×96 patches from full volumes | P0 |
| FR-P2 | Apply foreground-biased patch sampling (50% probability) | P1 |
| FR-P3 | Apply MONAI augmentation pipeline: flips, 90° rotations, intensity scaling/shift, Gaussian noise | P1 |
| FR-P4 | Provide separate train and validation transform compositions | P0 |
| FR-P5 | Use MONAI `list_data_collate` for batch collation | P0 |
| FR-P6 | Support `pin_memory` and `num_workers` for GPU-optimized loading | P1 |

### 7.3 Model Architectures
| ID | Requirement | Priority |
|---|---|---|
| FR-M1 | Implement Baseline 3D U-Net (feature widths 16/32/64/128) | P0 |
| FR-M2 | Implement Residual Attention 3D U-Net (feature widths 32/64/128/256) | P0 |
| FR-M3 | Integrate MONAI SwinUNETR with gradient checkpointing | P1 |
| FR-M4 | Expose model selection via `MODEL_NAME` config flag | P0 |
| FR-M5 | All models accept 4-channel input and output 3-class logits `[B, 3, D, H, W]` | P0 |

### 7.4 Training
| ID | Requirement | Priority |
|---|---|---|
| FR-T1 | Single-epoch training loop with mixed precision (`torch.cuda.amp`) | P0 |
| FR-T2 | Validation loop with sliding-window inference (overlap = 0.5) | P0 |
| FR-T3 | Save best (highest val Dice) and last checkpoint to `outputs/checkpoints/` | P0 |
| FR-T4 | Persist checkpoint metadata (epoch, val_dice, config) as sidecar JSON | P1 |
| FR-T5 | Log per-epoch metrics (train_loss, val_dice, learning_rate) to CSV | P0 |
| FR-T6 | Use `ReduceLROnPlateau` scheduler (factor=0.5, patience=5, mode=max) | P1 |
| FR-T7 | Use AdamW optimizer with weight decay 1e-2 | P1 |
| FR-T8 | Support configurable epochs (default: 75), batch size (default: 1), learning rate (default: 1e-4) | P0 |
| FR-T9 | Plot learning curves at end of training | P2 |

### 7.5 Loss Functions
| ID | Requirement | Priority |
|---|---|---|
| FR-L1 | Implement multi-class soft Dice loss | P0 |
| FR-L2 | Implement combined Dice + CrossEntropy loss (default) | P0 |
| FR-L3 | Implement combined Dice + Focal loss (selectable via flag) | P1 |
| FR-L4 | Support class-weighted CE: weights (0.2, 1.0, 2.0) | P1 |

### 7.6 Evaluation
| ID | Requirement | Priority |
|---|---|---|
| FR-E1 | Compute multiclass Dice Score (excluding background) on validation set | P0 |
| FR-E2 | Per-class Dice breakdown (WT, TC, ET) | P1 |
| FR-E3 | Generate confusion matrix for classification breakdown | P2 |
| FR-E4 | Export metric tables (CSV / Markdown) for the report | P1 |

### 7.7 Inference
| ID | Requirement | Priority |
|---|---|---|
| FR-I1 | Load a trained checkpoint from disk | P0 |
| FR-I2 | Run sliding-window inference on a full MRI volume (ROI 96³, overlap 0.25) | P0 |
| FR-I3 | Save predicted segmentation mask as NIfTI (preserving original affine) | P0 |
| FR-I4 | Optionally save per-class probability volumes as NIfTI | P1 |
| FR-I5 | Support single-case and CLI-driven batch inference | P0 |

### 7.8 Visualization & Reporting
| ID | Requirement | Priority |
|---|---|---|
| FR-V1 | 2D slice overlay visualization (FLAIR / T1ce / GT / prediction) | P1 |
| FR-V2 | 3D mesh export (`.obj` / `.stl`) of predicted tumor regions | P2 |
| FR-V3 | Select representative qualitative cases (best / median / worst) | P2 |
| FR-V4 | Generate training curves (loss / Dice / LR vs epoch) | P1 |
| FR-V5 | Confusion matrix visualization | P2 |

### 7.9 Configuration
| ID | Requirement | Priority |
|---|---|---|
| FR-C1 | Single centralized `configs/config.py` (no scattered magic numbers) | P0 |
| FR-C2 | Frozen `Config` dataclass exposing all settings | P0 |
| FR-C3 | Factory functions: `build_model`, `build_optimizer`, `build_loss` | P0 |

---

## 8. Non-Functional Requirements

### 8.1 Performance
| ID | Requirement |
|---|---|
| NFR-1 | Training step (forward + backward) for one 96³ patch ≤ 2 s on RTX 4050 |
| NFR-2 | Full-volume inference (240³ volume) ≤ 60 s on RTX 4050 |
| NFR-3 | Peak GPU memory during training ≤ 6 GB |
| NFR-4 | DataLoader keeps GPU saturated (GPU util > 80% during training) |

### 8.2 Reliability & Reproducibility
| ID | Requirement |
|---|---|
| NFR-5 | Fixed random seed (42) for all RNG sources (PyTorch, NumPy, CUDA) |
| NFR-6 | Deterministic validation metrics (no randomness in eval) |
| NFR-7 | Checkpointing enables exact training resumption |

### 8.3 Maintainability
| ID | Requirement |
|---|---|
| NFR-8 | Modular code structure: `datasets/`, `models/`, `training/`, `inference/`, `utils/`, `configs/` |
| NFR-9 | All modules have docstrings describing purpose and key arguments |
| NFR-10 | Centralized config — no hardcoded hyperparameters in scripts |

### 8.4 Portability
| ID | Requirement |
|---|---|
| NFR-11 | Runs on Windows / Linux / macOS (tested primarily on Windows) |
| NFR-12 | CPU fallback if CUDA unavailable (degraded performance) |
| NFR-13 | Small `NUM_WORKERS` (2) default for Windows compatibility |

### 8.5 Usability
| ID | Requirement |
|---|---|
| NFR-14 | Single-command training: `python -m training.train` |
| NFR-15 | Single-command inference: `python -m inference.predict --data_dir ... --checkpoint ... --out_dir ...` |
| NFR-16 | README documents setup, training, and inference steps |

### 8.6 Security & Compliance
| ID | Requirement |
|---|---|
| NFR-17 | No PHI handling — research dataset only |
| NFR-18 | No external network calls at runtime |

---

## 9. System Architecture

### 9.1 High-Level Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    MRI Scan (.nii.gz)                          │
│              [T1, T1ce, T2, FLAIR + Seg]                      │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              Dataset Loader  (datasets/brats_dataset.py)       │
│  • Canonical orientation                                       │
│  • Label remap {0,1,2,4} → {0,1,2,3}                           │
│  • Z-score normalization                                       │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│            MONAI Transforms  (datasets/dataloader.py)          │
│  • Spatial padding → 96³                                       │
│  • Random patch crop                                           │
│  • Augmentations: flips, rotations, intensity, noise           │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  PyTorch DataLoader                            │
│     pin_memory, num_workers=2, list_data_collate              │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│       Model  (models/unet3d.py or models/swinunetr.py)        │
│   BaselineUNet3D | ResidualUNet3D | SwinUNETR                  │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│        Loss  (configs/config.py)                              │
│   DiceLoss + CrossEntropyLoss  |  DiceLoss + FocalLoss        │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│      Optimizer + Scheduler (AdamW + ReduceLROnPlateau)         │
│        Mixed precision via torch.cuda.amp.GradScaler           │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│   Validation (sliding_window_inference + multiclass_dice)      │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│   Outputs: checkpoints (best.pt, last.pt) + metrics.csv       │
└────────────────────────────────────────────────────────────────┘
```

### 9.2 Module Structure

| Module | Responsibility |
|---|---|
| `configs/config.py` | Centralized hyperparameters, loss/optimizer factories, paths |
| `datasets/brats_dataset.py` | BraTS case discovery, NIfTI loading, label remap, normalization, patch sampling |
| `datasets/dataloader.py` | MONAI transform pipelines, DataLoader factories |
| `models/unet3d.py` | `BaselineUNet3D`, `UNet3D` (residual + attention), building blocks |
| `models/swinunetr.py` | MONAI SwinUNETR wrapper with checkpointing |
| `models/model_factory.py` | `build_model()` and `model_metadata()` dispatch by `MODEL_NAME` |
| `training/train.py` | Epoch loop, AMP training, validation, checkpointing, experiment logging |
| `inference/predict.py` | Single-case CLI inference, NIfTI mask + probability export |
| `inference/visualize_predictions.py` | 2D prediction overlays |
| `inference/generate_best_visualizations.py` | Curated qualitative outputs |
| `inference/visualize_3d.py` | 3D mesh rendering |
| `utils/metrics.py` | `multiclass_dice_score_3d` |
| `utils/brats_metrics.py` | BraTS-specific metric helpers |
| `utils/checkpoint_utils.py` | Checkpoint save/load utilities |
| `utils/experiment_logger.py` | CSV logger for training metrics |
| `utils/plot_metrics.py` | Loss / Dice / LR curve plotting |
| `utils/confusion_matrix.py` | Confusion matrix computation & visualization |
| `utils/select_visual_cases.py` | Best/median/worst case selection by Dice |
| `utils/export_qualitative.py` | Qualitative figure export |
| `utils/generate_metrics_table.py` | Markdown/CSV metric table generation |
| `utils/generate_visualizations.py` | Aggregated visualization pipeline |
| `utils/mesh_utils.py` | 3D mesh generation from masks |
| `utils/inference_utils.py` | Shared inference helpers |
| `utils/visualization.py` | Shared visualization helpers |

### 9.3 Data Flow Summary

1. **Training**: Volumes → BraTSDataset → patches (96³) → MONAI augmentations → DataLoader → Model → Loss → AdamW (AMP) → Checkpoint
2. **Validation**: Volume → BraTSDataset → padded → Sliding-window inference (96³, overlap 0.5) → Multiclass Dice
3. **Inference**: Volume → BraTSDataset → Sliding-window inference (96³, overlap 0.25) → Argmax → NIfTI mask (optionally + probability volumes)

---

## 10. Technical Specifications

### 10.1 Hardware Target
- **GPU**: NVIDIA RTX 4050, 6 GB VRAM (primary)
- **Fallback**: CPU (significantly slower; for debugging only)
- **RAM**: 16 GB minimum
- **Storage**: ≥ 50 GB free for BraTS dataset + checkpoints

### 10.2 Software Dependencies
- Python 3.10+
- PyTorch (CUDA-enabled build for GPU)
- MONAI (for transforms, SwinUNETR, sliding-window inference)
- NiBabel (NIfTI I/O)
- NumPy
- Matplotlib, scikit-learn, pyvista/matplotlib-3d (visualization)
- See `requirements.txt` and `requirements-visualization.txt` for full pinned list

### 10.3 Default Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Patch size | (96, 96, 96) | `PATCH_SIZE` |
| Batch size | 1 | `BATCH_SIZE` |
| Epochs | 75 | `NUM_EPOCHS` |
| Learning rate | 1e-4 | `LEARNING_RATE` |
| Optimizer | AdamW | `build_optimizer` |
| Weight decay | 1e-2 | `WEIGHT_DECAY` |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5, mode=max) | `train.py` |
| Mixed precision | True | `USE_MIXED_PRECISION` |
| Seed | 42 | `SEED` |
| Model | residual_unet | `MODEL_NAME` |
| Input channels | 4 (T1, T1ce, T2, FLAIR) | `INPUT_CHANNELS` |
| Output classes | 3 (WT, TC, ET) | `NUM_CLASSES` |
| CE class weights | (0.2, 1.0, 2.0) | `CE_CLASS_WEIGHTS` |
| Sliding window overlap (val) | 0.5 | `train.py` |
| Sliding window overlap (infer) | 0.25 | `inference/predict.py` |
| Num workers | 2 | `NUM_WORKERS` |
| Pin memory | True | `PIN_MEMORY` |

### 10.4 Dataset Layout
```
BraTS/
├── Training/
│   ├── case_00001/
│   │   ├── case_00001_t1n.nii.gz
│   │   ├── case_00001_t1c.nii.gz
│   │   ├── case_00001_t2w.nii.gz
│   │   ├── case_00001_t2f.nii.gz
│   │   └── case_00001_seg.nii.gz
│   ├── case_00002/ ...
├── Validation/
└── Testing/
```

### 10.5 Output Layout
```
brain-tumor-segmentation/
└── outputs/
    ├── checkpoints/
    │   ├── best.pt
    │   ├── best.json
    │   ├── last.pt
    │   └── last.json
    └── logs/
        ├── metrics.csv
        ├── loss_curve.png
        ├── dice_curve.png
        └── lr_curve.png
```

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU OOM on 6GB during training | Medium | High | Patch-based training (96³), AMP, gradient checkpointing for SwinUNETR |
| Slow training (large BraTS dataset) | High | Medium | Reduce patch size, lower resolution, fewer augmentations, mixed precision |
| Poor model convergence | Medium | High | Class-weighted loss, foreground-biased sampling, LR scheduling, multiple architecture options |
| Class imbalance (background >> tumor) | High | Medium | Dice + weighted CE loss; foreground-biased patch sampling |
| SwinUNETR unavailable / incompatible MONAI version | Low | Medium | Pin MONAI version; fall back to residual_unet |
| Windows multiprocessing issues | Medium | Low | `num_workers=2`, small worker count |
| Data leakage between train/val | Low | High | Train/val split at case level, never at patch level (already enforced) |
| NIfTI orientation mismatch | Medium | High | `nib.as_closest_canonical` for all volumes |

---

## 12. Milestones & Timeline

| Milestone | Status | Deliverable |
|---|---|---|
| M1: Data pipeline complete | ✅ Done | Dataset loader, MONAI transforms, DataLoaders |
| M2: Model architectures implemented | ✅ Done | Baseline 3D U-Net, Residual Attention U-Net, SwinUNETR |
| M3: Training loop complete | ✅ Done | AMP, scheduler, checkpointing, experiment logging |
| M4: Loss functions complete | ✅ Done | Dice, Dice+CE, Dice+Focal |
| M5: Inference pipeline complete | ✅ Done | Sliding-window inference, NIfTI export, CLI |
| M6: Metrics & evaluation | ✅ Done | Multiclass Dice, BraTS metrics, confusion matrix |
| M7: Visualization (2D + 3D) | ✅ Done | Slice overlays, 3D meshes, qualitative case selection |
| M8: Trained model + benchmark results | 🔄 In Progress | Run full 75-epoch training, report final Dice |
| M9: Final report & presentation | ⏳ Pending | Capstone report, demo slides |

---

## 13. Acceptance Criteria

The project will be considered complete when:

1. ✅ `python -m training.train` runs end-to-end without errors on the configured hardware.
2. ✅ Inference produces a valid NIfTI segmentation mask for a held-out case.
3. ⏳ Final validation Dice score is reported and meets the **Minimum Viable** tier (≥ 0.70).
4. ✅ All three model architectures (`baseline_unet`, `residual_unet`, `swinunetr`) can be selected and trained.
5. ✅ Qualitative visualizations (2D overlays and 3D meshes) are produced.
6. ✅ Experiment logs and checkpoints are persisted for reproducibility.
7. ✅ Codebase is documented via README and this PRD.

---

## 14. Future Work (Post-Capstone)

- **Cross-validation** training (5-fold BraTS standard).
- **Ensemble inference** across the three architectures.
- **nnU-Net baseline** comparison.
- **Test-time augmentation** for improved robustness.
- **Uncertainty estimation** via Monte Carlo dropout.
- **DICOM I/O wrapper** for clinical workflow integration.
- **Web-based viewer** (OHIF / 3D Slicer plugin).
- **Federated learning** across multi-institutional data.

---

## 15. Glossary

| Term | Definition |
|---|---|
| **BraTS** | Brain Tumor Segmentation challenge dataset |
| **WT / TC / ET** | Whole Tumor / Tumor Core / Enhancing Tumor — the three BraTS sub-regions |
| **FLAIR** | Fluid-Attenuated Inversion Recovery MRI sequence |
| **NIfTI** | Neuroimaging Informatics Technology Initiative — file format (`.nii`/`.nii.gz`) |
| **Dice Score** | Sørensen–Dice coefficient: `2·|A∩B| / (|A|+|B|)` — overlap metric for segmentation |
| **Hausdorff Distance** | Maximum distance between boundary points of two sets (boundary accuracy) |
| **AMP** | Automatic Mixed Precision (FP16 + FP32) for faster training |
| **Patch-based training** | Training on small 3D crops rather than full volumes (memory-efficient) |
| **Sliding-window inference** | Predicting on a full volume by sliding a small window with overlap |
| **SwinUNETR** | Swin Transformer + U-Net hybrid for 3D medical segmentation |
| **MONAI** | Medical Open Network for AI — PyTorch-based framework for medical imaging |

---

## 16. References

- BraTS Challenge: https://www.synapse.org/Synapse:syn53708249
- MONAI: https://monai.io/
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Hatamizadeh et al., "SwinUNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images" (2021)
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)

---

*End of Document*
