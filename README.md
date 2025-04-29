# TFG Extractor: Text-to-Excel Conversion for Scientific Papers

This project aims to convert plain text from scientific papers into structured Excel tables using various natural language processing (NLP) and deel learning techniques.

## Project Structure
- **data/**: Input texts.
- **core/**: Core modules common to all approaches.
- **tests/**: Unit and integration tests for each module.
- **traditional_bow/**: Traditional extraction using Bag of Words.

## Requirements
- Python 3.10-3.11 (Python 3.12+ are not fully soported by spaCy yet)
- pip
- It is recommended to use a virtual environment to avoid dependecy conflicts.

## Quick Installation

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

3. Install dependencies:
```bash
pip install -r requirements.txt
python setup.py
```

4. Run the main program to verify the installation:
```bash
python main.py
```

## Current Status
The project is in early stages of development.
