# 🧠 Brain Tumor Segmentation

This project focuses on **automated brain tumor segmentation** using deep learning on multi-modal MRI scans.

---

## 🚀 Complete Setup

### Prerequisites

- **Windows 10/11**
- **Python 3.11.x** (recommended - compatible with PyTorch and MONAI)
- **Node.js 18+**
- **npm**
- **NVIDIA GPU** (recommended for practical inference speed)
- **Git**

### 1. Clone Repository

```bash
git clone <repository-url>
cd brain-tumor-segmentation
```

### 2. Create Python Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install PyTorch (CUDA 12.1)

**For GPU (recommended):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only (slower):**
```bash
pip install torch
```

### 5. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 7. Configure Frontend API URL (Optional)

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` if you need to change the default API URL (defaults to `http://127.0.0.1:8000`):
```
VITE_API_URL=http://127.0.0.1:8000
```

### 8. Start Backend Server

From project root:

```bash
python -m uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 9. Start Frontend Development Server

Open a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

---

## 🔧 Model Requirements

The application expects a trained SwinUNETR checkpoint at:

```
outputs/exp_swinunetr_4class_et_fixed/checkpoints/best_mean_dice.pt
```

If you have a trained checkpoint, place it in the expected location. The inference pipeline will automatically detect and load it. The checkpoint is not included in requirements.txt due to size.

---

## 📊 Dataset Requirements

### Runtime Inference

**No dataset required for runtime inference.** 

Users only need to upload the four MRI modalities (T1, T1ce, T2, FLAIR) as NIfTI files through the web interface.

### Training (Optional)

If you want to train the model, the BraTS dataset should be organized as:

```
BraTS/
├── Training/
├── Validation/
└── Testing/
```

Training requires additional dependencies that are included in requirements.txt but are primarily used during the training process.

---

## 🎨 Class Visualization

The visualization uses the following color scheme:

- **NCR/NET** → Red
- **Edema** → Green  
- **Enhancing Tumor (ET)** → Purple/Magenta

---

## ⚙️ Tech Stack

**Backend:**
- Python 3.11
- PyTorch
- MONAI
- NiBabel
- NumPy
- SciPy
- FastAPI
- Uvicorn
- Pydantic

**Frontend:**
- React
- Vite
- NiiVue
- Tailwind CSS
- React Router
- Axios
- Lucide React

---

## 📁 Project Structure

```
brain-tumor-segmentation/
├── api/              # FastAPI backend
├── configs/          # Configuration files
├── datasets/         # Dataset loaders
├── inference/        # Inference scripts
├── models/           # Model definitions
├── training/         # Training scripts
├── utils/            # Utility functions
├── frontend/         # React frontend
└── outputs/          # Predictions and results
```

---

## 📌 Key Features

- **Multi-modal MRI processing** (T1, T1ce, T2, FLAIR)
- **4-class tumor segmentation** (Background, NCR/NET, Edema, Enhancing Tumor)
- **Real-time inference** via FastAPI
- **Interactive 3D visualization** with NiiVue
- **Per-class tumor analysis** with volume and dimensions
- **Collapsible class analysis cards**
- **Ground truth validation** for research purposes

---

## 🎯 Project Overview

The goal of this project is to build an AI system that can:

* Segment brain tumors from MRI scans
* Support clinical decision-making
* Reduce manual effort in medical image analysis

The system uses a **SwinUNETR architecture** with MONAI and processes **multi-modal MRI data (T1, T1ce, T2, FLAIR)**.

---

## 👨‍💻 Author

Final Year Computer Science Project
AI-Assisted Brain Tumor Segmentation
