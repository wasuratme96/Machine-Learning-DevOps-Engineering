# (Hands-On) Machine Learning Data Pipeline

## Project Description
![Project Overview](./Asset/ProjectOverview.png) <br/>
This project will bring you go throuhgt on first element of Machine Learning in MLOps which is ```Data Pipeline``` <br/>
In every machine learning project, data pipeline is mandatory to make our incoming data ready to use, alidate data profile and finding some insight from new data

## Set Up 
Before your start to follow along this tutorial, what you have to install are **WeightAndBias**, **Conda**, **MLFlow** and **Hydra**

**[Weight and Bias](https://wandb.ai/home)** <br/>
Weight and Bias is free online service for xxxxxxxx
``` %python
pip install wandb
```

**[Conda](https://docs.conda.io/en/latest/)** <br/>
Conda is python environment management library.
In every pieces of machine learning pipelie (ex. dowload_data, preprocessing) We will use Conda to encapsulate all requirement for running the script on each pipline.
``` %python
pip install conda
```

**[MLFlow](https://mlflow.org/)** <br/>

``` %python
pip install mlflow
```
**[Hydra](https://github.com/facebookresearch/hydra)** <br/>
Hydra is frame for hierarchical configuration creation. This library enable you to store complex config variables and override it via config file and command line.
``` %python
pip install hydra-core --upgrade
```


After you install weight and bias on your local machine and completed account creation on [website](https://wandb.ai) <br/>
To get API key for authentications on communication from your local machine to weight and bias service, please use this link [https://wandb.ai/authorize](https://wandb.ai/authorize) and use below command in CLI
``` %python
wand longin [API key]
```

All exploratory data analysis and model development have been experimented  in **Jupyter Notebook** and convert into *.py* format for modularlization and testing purpose.


## Purpose of Pipeline
This project use data from [kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers) which contains data about customer infomations total 21 columns and 10,000 rows. 2 Group of data are Demographic data of customer and Bank related data.

**Date**
- Attrition_Flag : Exiting Customer/ Attrited Customer


## File Structure
    └── ML Data Pipeline HandsOn
        ├── conda.yml             
        ├── config.yml
        ├── main.py
        ├── MLProject
        ├── download           # Collect all images results
        |   ├── conda.yml            
        |   └── results       
        ├── logs                   # Collect all logs from running
        ├── models                 # Path to store trained ml models
        ├── churn_notebook.ipynb   # Notebook for experiment
        ├── churn_library.py                   # Main script for running
        ├── churn_script_logging_and_tests.py  # Test and logging script
        ├── settings.py                        # Collect constant values
        ├── requirement.txt                    # List of require library versions
        └── README.md                          # This file