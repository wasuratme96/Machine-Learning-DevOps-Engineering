# (Hands-On) Machine Learning Data Pipeline

## Project Description
![Project Overview](./Asset/ProjectOverview.png)

## Set Up 
Before your start to follow along this tutorials, what you have to install are WeightAndBias, Conda and MLFlow



All exploratory data analysis and model development have been experimented  in **Jupyter Notebook** and convert into *.py* format for modularlization and testing purpose.
![Correlation Heat Map](./Asset/Production%20Code%20Ready%20DS.png)

## Data
This project use data from [kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers) which contains data about customer infomations total 21 columns and 10,000 rows. 2 Group of data are Demographic data of customer and Bank related data.

**Target Columns:**
- Attrition_Flag : Exiting Customer/ Attrited Customer


## File Structure
    └── ML Data Pipeline Hands-On
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
        ├── requirement.txt                    # List of require library versions
        └── README.md                          # This file