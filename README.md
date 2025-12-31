# Benchmarking Deep Learning Models for Raman Spectra Classification

This repository provides a unified benchmarking framework for evaluating multiple deep learning–based Raman spectra classifiers on **three open-source Raman spectroscopy datasets**:

- **MLROD**
- **Bacteria-ID**
- **API (Active Pharmaceutical Ingredients)**

The following models are benchmarked under consistent training and evaluation protocols:

- **Deep CNN** (referred to as `mlrod` in the codebase)
- **SANet**
- **RamanNet**
- **Transformer**
- **RamanFormer**

All models are implemented in **PyTorch**, and the pipeline supports dataset preprocessing, training, hyperparameter tuning, and evaluation using **test accuracy** and **macro-averaged F1 score**.

## Setup Instructions
Follow the steps below to reproduce the benchmarking experiments.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>

aaa
