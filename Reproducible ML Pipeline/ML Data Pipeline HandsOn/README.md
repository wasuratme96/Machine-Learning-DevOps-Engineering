# (Hands-On) Machine Learning Data Pipeline

## Project Description
![Project Overview](./Asset/ProjectOverview.png) <br/>
This project will bring you go throuhgt on first element of Machine Learning in MLOps which is ```Data Pipeline``` <br/>
In every machine learning project, data pipeline is mandatory to make our incoming data ready to use, alidate data profile and finding some insight from new data

## Contents
- [Environment Setup](#set-up)
- [Purpose of Pipeline](#purpose-of-pipeline)
- [File Structure](#file-structure)
- [File Structure](#file-structure)
  * [Fork the Starter Kit](#fork-the-starter-kit)

  
## Environment Setup
Before your start to follow along this tutorial, what you have to install are  **Conda**, **MLFlow** and **Hydra** and **WeightAndBias**

**[Conda](https://docs.conda.io/en/latest/)** <br/>
Conda is python environment management library.
In every pieces of machine learning pipelie (ex. dowload_data, preprocessing) We will use Conda to encapsulate all requirement for running the script on each pipline.
```bash
> pip install conda
```
Then create conda environment with ```enviroment.yml```

``` bash
> conda env create -f environment.yml
> conda activate wandb_datapipeline
```

**[MLflow](https://mlflow.org/)** <br/>
MLflow provide you a bunch of lightweight APIs for machine learning life cycle development such as experiment tracking, packing code as reproducible pipeline and deployment.<br/>
In this tutorials we will use [MLflow Projects](https://www.mlflow.org/docs/latest/projects.html) to pack our python script into resuable date pipeline.

**[Hydra](https://github.com/facebookresearch/hydra)** <br/>
Hydra is frame for hierarchical configuration creation. This library enable you to store complex config variables and override it via config file and command line.

**[Weight and Bias](https://wandb.ai/home)** <br/>
Weight and Bias is machine learning platform that come with many features including **tracking**, **versioning data and model**, **data visualization**, **reporting** and **hyperparameters sweeper**. <br/>
![weightandbias_overview](./Asset/weightandbias_overview.gif) <br/>
After you install weight and bias on your conda environment from previous step and completed account creation on [website](https://wandb.ai) <br/>
To get API key for authentication on communication from your local machine to weight and bias service, please use this link [https://wandb.ai/authorize](https://wandb.ai/authorize) and use below command in CLI
``` %python
wand longin [API key]
```

On this tutorial, we will use WandB to store our processed data and track all the script run.



## Purpose of Pipeline
In this tutorials we use data from [Sklearn](https://www.kaggle.com/sakshigoyal7/credit-card-customers) which contains data about customer infomations total 21 columns and 10,000 rows. 2 Group of data are Demographic data of customer and Bank related data.

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