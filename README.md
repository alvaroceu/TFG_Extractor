# TFG Extractor: Text-to-Excel Conversion for Scientific Papers

This project aims to convert plain text from scientific papers into structured Excel tables using various natural language processing (NLP) and deep learning techniques.

## Project Structure
- **data/**: Databases and results.
- **core/**: Core modules common to all approaches.
- **tests/**: Unit and integration tests for each module.
- **traditional_bow/**: Traditional extraction using Bag of Words and TF-IDF
- **traditional_embeddings/**: Traditional extraction using gloVe.
- **LSTM_methods**: Advanced extraction using USE-DAN.
- **transformer_method/**: TState of art extraction using DistilBERT, BERTLarge, and 2 proposed arquitectures (Sparse/BiL).
- **trained_models/**: Weights for the BiL models, in both a frozen and complete training.

## Requirements
- Python 3.13+
- pip
- It is recommended to use a virtual environment to avoid dependecy conflicts.

## Installation and setup
See [INSTALL.md](INSTALL.md) for detailed setup instructions, including how to:
- Set up virtual environment
- Install dependencies
- Download required resources
- Run de setup script

## Current Status
The project is finished, having succesfully developed 2 variations of the BERT architecture. BiL architecture is available at [Hugging Face](https://huggingface.co/alvaroceu/BiLBERT-Large-QA-SQuAD2)

## Author
Created by **Álvaro Sánchez Mateos** as part of his Bachelor's Thesis at Universidad CEU San Pablo, with the support of his tutors: **ÁAna Sanmartín Domenech** and **Guillermo de la Calle Velasco**
