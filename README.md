# Benchmarking Deep Learning Models for Raman Spectra Classification

This repository provides a unified benchmarking framework for evaluating multiple deep learning–based Raman spectra classifiers on **three open-source Raman spectroscopy datasets**:

- **MLROD <sup> [1] </sup>**
- **Bacteria-ID <sup> [2] </sup>**
- **API (Active Pharmaceutical Ingredients) <sup> [3] </sup>**

The following models are benchmarked under consistent training and evaluation protocols:

-**Support Vector Classifier**
-**Random Forest**
- **Deep CNN <sup> [4] </sup>** 
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
Create the following empty directory
```bash
mkdir datasets/Bacteria_ID
```

### 3. Download the MLROD dataset
Download it from https://odr.io/MLROD#/search/display/1348/eyJkdF9pZCI6IjYwMCJ9 and place it in the directory `datasets/MLROD/`. <br> <br>
Run the `processing.ipynb` and `test_processing.ipynb` notebooks to generate .pkl files containing all the spectra interpolated to have a common spectral domain.

### 4. Download the Bacteria_ID dataset 
Download it from https://github.com/csho33/bacteria-ID/blob/master/README.md and place it in the directory `datasets/Bacteria_ID`.

### 5. Download the API dataset
Download it from https://springernature.figshare.com/articles/dataset/Open-source_Raman_spectra_of_chemical_compounds_for_active_pharmaceutical_ingredient_development/27931131 and place it in the directory `datasets/Pharma/`. <br> <br>
Run the `explore.ipynb` notebook to generate the .pkl files for the train, validation and test splits of the dataset.

## Training Models
### 1. Finding the optimum hyperparameters
To find the optimum hyperparameters for a specific model on a specific dataset, execute the corresponding training module located in the `train/hyperparameter_tuning` directory as a Python script. For example, to find the optimum hyperparameters for the RamanNet model on the Bacteria ID dataset for the 30 category isolate classification problem, run the following from the root directory

```bash
python -m train.hyperparameter_tuning.Bacteria_ID.thirty.train_RamanNet
```
Hyperparameter tuning is performed using the validation set. The model achieving the best validation accuracy is saved to `results/hyperparameter_tuning/trained_models/`.

### 2. Running multiple trials for statistical evaluation
To obtain the final results for a specific model on a specific dataset across multiple runs, execute the corresponding training module located in the  `train/final_multi_run/` directory as a Python script. For example, to train the RamanNet model on the Bacteria ID dataset for the 30 category isolate classification problem using a learning rate of 0.001, batch size of 32 and for 5 runs, execute the following from the root directory

```bash
python -m train.final_multi_run.Bacteria_ID.thirty.train_RamanNet --batch_size 32 --learning_rate 0.001 --runs 5
```
The default value for these arguments are the optimal hyperparameters that we obtained. The model achieving the best validation accuracy is saved to `results/final_multi_run/trained_models/`.


## Evaluation 
To compute test accuracy and macro F1 score aggregated across runs, execute the corresponding evaluation notebooks located in `results/analysis/acc_calc/trained_models/`. <br> <br>
Ensure that the paths to the trained model checkpoints are updated appropriately before running the notebooks.

## Citations
<ol>
  <li>Berlanga, Genesis, Quentin Williams, and Nathan Temiquel. "Convolutional neural networks as a tool for Raman spectral mineral classification under low signal, dusty Mars conditions." Earth and Space Science 9.10 (2022): e2021EA002125.</li>
  <li>Ho, Chi-Sing, et al. "Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning." Nature communications 10.1 (2019): 4927.</li>
  <li>Flanagan, Aaron R., and Frank G. Glavin. "Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development." Scientific Data 12.1 (2025): 498.</li>
  <li>Liu, Jinchao, et al. "Deep convolutional neural networks for Raman spectrum recognition: a unified solution." Analyst 142.21 (2017): 4067-4074.</li>
  <li>Deng, Lin, et al. "Scale-adaptive deep model for bacterial raman spectra identification." IEEE Journal of Biomedical and Health Informatics 26.1 (2021): 369-378.</li>
  <li>Ibtehaz, Nabil, et al. "RamanNet: a generalized neural network architecture for Raman spectrum analysis." Neural Computing and Applications 35.25 (2023): 18719-18735.</li>
  <li>Liu, Bo, et al. "Classification of deep-sea cold seep bacteria by transformer combined with Raman spectroscopy." Scientific Reports 13.1 (2023): 3240.</li>
  <li>Koyun, Onur Can, et al. "RamanFormer: A transformer-based quantification approach for Raman mixture components." ACS omega 9.22 (2024): 23241-23251.</li>
</ol>
