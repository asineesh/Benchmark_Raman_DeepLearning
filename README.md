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


Evaluation

To compute test accuracy and macro-averaged F1 score:

Navigate to:

results/trained_models/


Open the corresponding evaluation notebook

Update the file paths to point to the trained model checkpoints

Run the notebook to obtain final performance metrics
