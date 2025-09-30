The layout of the repository is as follows:
  -
  -  data

    Scripts for processing the raw raw data into Pytorch-ready training/testing datasets
    Analysis
    
    scripts and notebooks for analyzing Training, testing, and Oktobefest results.
  
  -  torch

     All code for model training and testing.

Clone the code
  -
  git clone

Annotate Raw data
  -

  - Go into data directory
  
    cd data

  - Run annotation script to create annotated dataset of PSMs
  
    python test_annotation.py

Create training set for model
  -
  - Set configuration files for desired annotation settings
  - Create training dataset from annotated search data
    
    python merge_search_into_training_data.py

  - Create your token dictionary
    
    python enumerate_tokens.py {traing_set_data_directory}

Train the model
  -
  - Go into torch directory
    
    cd torch
  
  - Set all configuration files to appropriate settings for model, data, training
  - Run training

    python main.py

Test the model
  -
  - Run main with any argument

    python main.py {any_argument}
    
