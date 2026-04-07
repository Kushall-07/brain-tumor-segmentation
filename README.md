# 🧠 Brain Tumor Segmentation

This project focuses on **automated brain tumor segmentation** using deep learning on multi-modal MRI scans.

---

## 🚀 Project Overview

The goal of this project is to build an AI system that can:

* Segment brain tumors from MRI scans
* Support clinical decision-making
* Reduce manual effort in medical image analysis

The system uses a **3D U-Net architecture** and processes **multi-modal MRI data (T1, T1ce, T2, FLAIR)**.

---

## 🧩 Work Completed So Far

### ✅ 1. Project Configuration

* Created a centralized `config.py` file
* Handles:

  * Device selection (CPU/GPU)
  * Hyperparameters (batch size, learning rate, epochs)
  * Dataset paths
  * Mixed precision settings

---

### ✅ 2. Dataset Loader (BraTS MRI)

* Built a custom PyTorch dataset:

  * Loads `.nii/.nii.gz` MRI files using NiBabel
  * Supports 4 modalities:

    * T1, T1ce, T2, FLAIR
  * Stacks them into:

    ```
    [4, Depth, Height, Width]
    ```
* Ensures correct 3D orientation
* Applies **z-score normalization**

---

### ✅ 3. Label Preprocessing

* Fixed BraTS label format:

  ```
  Original: 0, 1, 2, 4
  Converted: 0, 1, 2, 3
  ```
* Ensures compatibility with deep learning models

---

### ✅ 4. Data Pipeline with MONAI

* Implemented a robust preprocessing pipeline using MONAI:

  * Spatial padding (prevents crashes)
  * Patch extraction (96×96×96)
  * Random augmentations:

    * Flips
    * Rotations
    * Intensity scaling
    * Noise injection

---

### ✅ 5. Patch-Based Training Setup

* Instead of full MRI volumes:

  ```
  Input → [4, 96, 96, 96]
  ```
* Benefits:

  * Fits GPU memory (RTX 4050 - 6GB)
  * Faster training
  * Better generalization

---

### ✅ 6. DataLoader Integration

* Created modular loaders:

  * `get_train_loader()`
  * `get_val_loader()`
* Features:

  * GPU-optimized loading (`pin_memory`)
  * Parallel loading (`num_workers`)
  * MONAI-compatible batching

---

## 🧠 Current Pipeline

```
MRI Scan (.nii)
      ↓
Dataset Loader
      ↓
Label Remapping
      ↓
Normalization
      ↓
MONAI Transforms
      ↓
Patch Extraction (96³)
      ↓
DataLoader
      ↓
Ready for Model Training 🚀
```

---

## ⚙️ Tech Stack

* Python
* PyTorch
* MONAI
* NiBabel
* NumPy

---

## 📌 Next Steps

* Implement **3D U-Net model**
* Build training loop
* Add Dice score evaluation
* Perform inference and visualization

---

## 🎯 Status

🟢 Data pipeline complete
🟡 Model implementation pending
🔜 Training phase next

---

## 👨‍💻 Author

Final Year Computer Science Project
AI-Assisted Brain Tumor Segmentation
