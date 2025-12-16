# Prosit MultiFrag: A prosit model for multiple fragmentation methods
## Introduction
Prosit MultiFrag is a recurrent neural network trained jointly on 5 different types of fragmentation spectra -> HCD, ECD, EID, UVPD, and ETciD. The model outputs 815 ion types, which include ion series a, a+proton, b, c, c+proton, x, x+proton, y, z, z+proton, up to length 29 and product charge +3.

The original model, published on Koina, was trained on ~2.1 million unique PSMs, obtained through MSFragger searches, roughly equally divided between the 5 fragmentation types. Each fragmentation type was run on digests using 5 different enzymes: LysN, LysC, GluC, Trypsin, and Chymotrypsin. The instrument used was an Orbitrap Exploris (Thermo Fisher Scientific) equipped with an Omnitrap (Fasmatech). The raw files were provided by Dr. Shabaz Mohammed of The University of Oxford, Oxford, England. Project data can be found in `https://zenodo.org/records/15755223`

This repository provides all relevant code for recapitulating the project, from data processing into training ready datasets, to model training and evaluation.

## Directory layout
- `data/` - Annotation, data processing, and training set creation scripts
  - `yaml/`
    - `annotate.yaml` - Settings for how to annotate raw data
    - `create_dataset.yaml` - Settings for how to create training data
  - `enumerate_tokens.py` - Utilities for tokenizing modified sequences and determining token dictionary
  - `mass_scale.py` - Utilities for calculating fragment masses and annotation
  - `merged_search_into_training_data.py` - Turning annotation results (all_psms.py) into a Pytorch/Huggingface ready training dataset
  - `test_annotation.py` - Script for annotating raw data with search results
- `torch/` - Model training and testing code
  - `models/`
    - `model_parts.py` - Layers for building peptide encoder
    - `peptide_encoder.py` - Transformer model
    - `prosit.py` - RNN model
  - `yaml`
    - `master.yaml` - Main settings for running training
    - `loader.yaml` - Settings for datasets and dataloading filters
    - `eval.yaml` - Settings for evaluation
    - `model.yaml` - Architectural settings for model
  - `loader_hf.py` - Code for Huggingface dataset/loader
  - `main.py` - Main script for running model training and testing
  - `losses.py` - Code for training loss and evaluation functions
  - `utils.py` - Miscellaneous utilities
## Usage
### Clone the project
```
git@github.com:wilhelm-lab/Prosit_multifrag.git
```
### Create a conda environment
Libraries necessary to run code in the repository
- `torch` - Model training and deployment
- `datasets` - Huggingface datasets and utilities
- `wandb` - Model training monitoring
- `pandas` - Used throughout project
- `numpy` - Used throughout project
- `tqdm` - Used throughout project
- `oktoberfest` - Utilities for processing raw data
- `yaml` - Used throughout project
### Annotate the raw data
- Enter `data` directory
```
cd data
```
- Set configuration settings in `yaml/annotate.yaml`
- Run annotation for all fragmentation methods
  - Must run the script once for every annotation method
```
python test_annotation.py
```
- Outputs annotation results in file named `all_psms.parquet` - one for each fragmenation method
### Create training data for model
- Enter `data` directory
```
cd data
```
- Set configuration settings in `yaml/create_dataset.yaml`
- Run script to create training parquet files from annotation results
```
python merged_search_into_training_data.py
```
- Get token dictionary by enumerating tokens in resulting training files
```
python enumerate_tokens.py {traing_set_data_directory}
```
### Train the model
- Go into torch directory
```
cd torch
```
- Set all configuration files to appropriate settings for model, data, training
- Run training
```
python main.py
```
### Test the model
- Go into `torch` directory
```
cd torch
```
- Set configuration settings in `eval.yaml`
- Run main script with any argument
```
python main.py {any argument}
```
- Outputs a parquet results file
---
Publication (pre-print): [MultiFrag pre-print](https://www.biorxiv.org/content/10.1101/2025.05.28.656555v1)
---
For questions, please contact *joel.lapin@tum.de*
