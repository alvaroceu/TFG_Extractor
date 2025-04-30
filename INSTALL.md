# Installation Guide for TFG Extractor

This guide will jelp you set up the TFG Extractor project locally, including environment setup, dependency installation, and required NLP resources.

## Prerequisites
- **Python 3.13+**
- **pip** - Python package installer.
- (Optional but recommended) **virtualenv** for environment isolation.

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
4. Install dependencies and download resources:
```bash
pip install -r requirements.txt
python setup.py
```
5. Run the main program to verify the installation:
```bash
python main.py
```