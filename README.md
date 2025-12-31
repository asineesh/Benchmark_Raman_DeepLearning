# Benchmarking Deep Learning Models for Raman Spectra Classification

This repository provides a unified benchmarking framework for evaluating multiple deep learning–based Raman spectra classifiers on **three open-source Raman spectroscopy datasets**:

- **MLROD <sup> [1] </sup>**
- **Bacteria-ID <sup> [2] </sup>**
- **API (Active Pharmaceutical Ingredients) <sup> [3] </sup>**

The following models are benchmarked under consistent training and evaluation protocols:

- **Deep CNN <sup> [4] </sup>** (referred to as `mlrod` in the codebase)
- **SANet <sup> [5] </sup>**
- **RamanNet <sup> [6] </sup>**
- **Transformer <sup> [7] </sup>**
- **RamanFormer <sup> [8] </sup>**

All models are implemented in **PyTorch**, and the pipeline supports dataset preprocessing, training, hyperparameter tuning, and evaluation using **test accuracy** and **macro-averaged F1 score**.

## Setup Instructions
Follow the steps below to reproduce the benchmarking experiments.

### 1. Clone the Repository
```bash
git clone https://github.com/asineesh/Benchmark_Raman_DeepLearning/
cd Benchmark_Raman_DeepLearning
```
### 2. Create Required Directory Structure
Create the following empty directories
```bash
mkdir results/Bacteria_ID/models
mkdir datasets/Bacteria_ID
```

### 3. Download the MLROD dataset
Download it from https://odr.io/MLROD#/search/display/1348/eyJkdF9pZCI6IjYwMCJ9 and place it in the directory datasets/MLROD. <br> <br>
Run the `processing.ipynb` and `test_processing.ipynb` notebooks to generate .pkl files containing all the spectra interpolated to have a common spectral domain.

### 4. Download the Bacteria_ID dataset 
Download it from https://github.com/csho33/bacteria-ID/blob/master/README.md and place it in the directory `datasets/Bacteria_ID`.

### 5. Download the API dataset
Download it from https://springernature.figshare.com/articles/dataset/Open-source_Raman_spectra_of_chemical_compounds_for_active_pharmaceutical_ingredient_development/27931131 and place it in the directory `datasets/Pharma/`. <br> <br>
Run the `explore.ipynb` notebook to generate the .pkl files for the train, validation and test splits of the dataset.

## Training Models
To train a model on a given dataset, execute the corresponding training module located in the `results/ directory` as a Python script. 
During training: 
<ul>
  <li>Hyperparameter tuning is performed using the validation set.</li>
  <li>The model achieving the best validation accuracy is saved to `results/trained_models/` </li>
</ul>

## Evaluation 
To compute test accuracy and macro F1 score, run the corresponding evaluation notebooks located in `results/trained_models/`. <br> <br>
Ensure that the paths to the trained model checkpoints are updated appropriately before running the notebooks.
