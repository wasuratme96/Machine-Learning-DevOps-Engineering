# (Hands-On) Machine Learning Data Pipeline

## Overview
![Project Overview](./Asset/ml_Data_pipe_overview.png) <br/>
This project will bring you go throuhgt on first element of Machine Learning in MLOps which is ```Data Pipeline``` <br/>
In every machine learning project, data pipeline is mandatory to make our incoming data ready to use(train-test split), validate data profile and finding some insight from new data. <br/>
Step-by-Step simple re-usable data pipeline creation with MLflow and Weight and Bias plarform will be introduced here.


## Contents
- [Environment Setup](#set-up)
- [Data](#data)
- [Purpose of Pipeline](#purpose-of-pipeline)
- [File Structure](#file-structure)
- [Main](#main)
  * [ML Project File](#ml-project-file)
  * [Hydra Config File](#hydra-config-file)
  * [Conda File](#hydra-config-file)
- [Download Component](#download_component)
  * [ML Project File](#ml-project-file)
  * [Python Script](#python-script)
- [Preprocess Component](#preprocess_component)
  * [ML Project File](#ml-project-file)
  * [Python Script](#python-script)
- [CheckData Component](#checkdata_component)
  * [ML Project File](#ml-project-file)
  * [Python Script](#python-script)
- [Segregate Component](#segregate_component)
  * [ML Project File](#ml-project-file)
  * [Python Script](#python-script)
  
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

We will store all configurable variables in ```config.yml``` and use all variable via arguments parser when MLflow call the pipeline.

**[Weight and Bias](https://wandb.ai/home)** <br/>
Weight and Bias is machine learning platform that come with many features including **tracking**, **versioning data and model**, **data visualization**, **reporting** and **hyperparameters sweeper**. <br/>
![weightandbias_overview](./Asset/weightandbias_overview.gif) <br/>
After you install weight and bias on your conda environment from previous step and completed account creation on [website.](https://wandb.ai) <br/>
To get API key for authentication on communication from your local machine to weight and bias service, please use this link [https://wandb.ai/authorize](https://wandb.ai/authorize) and use below command in CLI
``` %python
wand longin [API key]
```

On this tutorial, we will use weight and bias to store processed data and track all the script run.

## Data
In this tutorials we use songs dataset from [Kaggle](https://www.kaggle.com/mrmorj/dataset-of-songs-in-spotify). Data set is collected by Spotify API.<br/>


This data contains sound chraracteristics (ex. danceability, loundness, valence) from on song with its genre. All of them are numerical value.

End goal is to classify song genre with sound characteristics data. (Multi-Class Classification)

## Purpose of Pipeline


### **Steps in Pipeline**
**dowload**
> 1.) Download raw data from github url <br/>
> 2.) Upload raw data as artifact into Weight and Bias <br/>

**preprocess**
> 1.) Upload raw data as artifact into Weight and Bias <br/>
> 2.) Upload raw data as artifact into Weight and Bias <br/>

**checkdata**
> 1.) Upload raw data as artifact into Weight and Bias <br/>
> 2.) Upload raw data as artifact into Weight and Bias <br/>
> 3.) Upload raw data as artifact into Weight and Bias <br/>

**segregate**
> 1.) Upload raw data as artifact into Weight and Bias <br/>
> 2.) Upload raw data as artifact into Weight and Bias <br/>
> 3.) Upload raw data as artifact into Weight and Bias <br/>
> 4.) Upload raw data as artifact into Weight and Bias <br/>


## File Structure
    └── ML_Data_Pipeline
        ├── config.yml      # Hydra config file
        ├── conda.yml             
        ├── main.py
        ├── MLProject
        |
        ├── download        # Download component
        |   ├── conda.yml            
        |   ├── MLproject
        |   └── main.py
        ├── preprocess      # Preprocess component
        |   ├── conda.yml            
        |   ├── MLproject
        |   └── main.py
        ├── checkdata       # Checkdata component
        |   ├── conda.yml            
        |   ├── MLproject
        |   └── main.py
        └── segregate       # Segregate component
            ├── conda.yml            
            ├── MLproject
            └── main.py
 