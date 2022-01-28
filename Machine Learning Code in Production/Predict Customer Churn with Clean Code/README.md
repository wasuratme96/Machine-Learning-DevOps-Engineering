# Predict Customer Churn with Production Ready Code

## Project Description
Main purpose of this project is to create machine learning model script that can **predict credit card customers that are most likely to churn** with PEP8 and **software engineering best pratices** for implementing machine learning in production (modular, documented and tested)

All EDA and model development have been experimented  in **Jupyter Notebook** and convert into *.py* format for modularlization and testing purpose.

## Data
This project use data from [kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers) which contains data about customer infomations total 23 columns and 10,000 rows. 

Target Columns:
- Attrition_Flag : Exiting Customer/ Attrited Customer

Demographic Data:
- Gender : M (Male), Female
- 


## Project Structure
    
    └── Predict Customer Churn with Clean Code 
        ├── data             # Folder to store data
        |   └── data profile # Folder to store profile of data
        ├── images           # Collect all images results
        |   ├── eda            
        |   └── results       
        ├── logs                   # Collect all logs from running
        ├── models                 # Path to store trained ml models
        ├── churn_notebook.ipynb   # Notebook for experiment
        ├── churn_library.py                   # Main script for running
        ├── churn_script_logging_and_tests.py  # Test and logging script
        ├── settings.py                        # Collect constant values
        └── README.md                          # This file

## Running Files
Main file to run is ```churn_library.py```. This will contains all functions for EDA, model training and save all results from EDA and trained model into target directory.<br/>

Thse package can interactively run from common-line interface (CLI) by follow script

``` CLI
python churn_library.py
```
For unit testing all functions, you can use ```churn_script_logging_and_tests.py```
``` CLI
python churn_script_logging_and_tests.py
```

## Process In 

