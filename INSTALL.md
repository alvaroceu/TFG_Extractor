# Installation Guide for TFG Extractor

This guide will jelp you set up the TFG Extractor project locally, including environment setup, dependency installation, and required NLP resources.

## Prerequisites

### System Requirements
- **Python 3.13+**
- **pip** - Python package installer.
- (Optional but recommended) **virtualenv** for environment isolation.

### For GPU Support (PyTorch with CUDA)
- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **CUDA Toolkit** (12.4 or 12.6 recommended)
- **Microsoft Visual C++ Redistributable** (required for PyTorch on Windows)
  - Download: https://aka.ms/vs/16/release/vc_redist.x64.exe
  - This is a **system dependency**, not installed via pip

If you lack these, you can use the **CPU version** of PyTorch (slower but works on any system).

## Step by step installation

1. Clone the repository:
```bash
git clone https://github.com/alvaroceu/TFG_Extractor
cd tfg_extractor
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # Linux
venv\Scripts\activate # Windows
```
3. Upgrade core tools (Optional but recommended):
```bash
pip install --upgrade pip setuptools wheel
```

4. Install dependencies:

**Option A: GPU with CUDA (recommended if you have NVIDIA GPU)**
```bash
# First, install PyTorch for your CUDA version manually
# CUDA 12.4:
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# OR CUDA 12.6:
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Then install remaining dependencies (in both cases)
pip install -r requirements.txt
```

**Option B: CPU only (simpler, works everywhere)**
```bash
pip install torch ==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

5. Download NLP resources:
```bash
python setup.py
```

6. Download Nused databases:
```bash
python data\load_data.py
```

7. Run the main program to verify the installation:
```bash
python main.py
```