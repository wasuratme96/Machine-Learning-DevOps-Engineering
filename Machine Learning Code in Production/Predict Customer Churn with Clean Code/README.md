# Predict Customer Churn with Production Ready Code

## Project Description
Main purpose of this project is to create machine learning model that can **predict credit card customers that are most likely to churn** with PEP8 and **software engineering best pratices** for implementing machine learning in production (modular, documented and tested)

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
        ├── images           # Collect all images results
        |   ├── eda            
        |   ├── results       
        ├── logs                   # Collect all logs from running
        ├── models                 # Path to store trained ml models
        ├── churn_notebook.ipynb   # Notebook for experiment
        ├── churn_library.py                   # Main script for running
        ├── churn_script_logging_and_tests.py  # Test and logging script
        └── README.md                          # This file

## Running Files
2 Main file to use and test script are ```churn_library.py``` and ```churn_script_logging_and_tests.py```. <br/>
Thse package can interactively run from common-line interface (CLI) by follow script

``` CLI
python churn_library.py
python churn_script_logging_and_tests.py
```


