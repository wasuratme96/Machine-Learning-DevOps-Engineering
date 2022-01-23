'''
Use to predict churn of bank customer with binary classification problem by
using selected machine learning model from Sklearn.

Exploratory data analysis and model tuning with gridsearch method also are included.

Usage : ./churn_library.py
Authors : Wasurat Soontronchai <wasurat_me96@outlook.com>
'''


# Standard Libary
import joblib
import logging

# Data Manipulations Library
import pandas as pd
import numpy as np

# Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Machine Learning Related Libary
import shap
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

# Settings and Constant
import settings

logging.basicConfig(
        filename = './logs/results.log',
        level = logging.INFO,
        filemode = 'w',
        format = '%(name)s - %(levelname)s - %(message)s'
)

def import_data(file_path: str,
                target_column: str) -> pd.DataFrame:
    '''
    Returns dataframe for the csv given from file path.
    And remove unrelated columns due to sd
    Args:
            file_path (str): A path to the csv file
            target_column (str) : Columns name of target prediction
    Outsputs:
            dataframe (pd.DataFrame): pandas dataframe
    '''
    try:
            assert isinstance(file_path, str)

            dataframe = pd.read_csv(file_path, index_col = 0)
            logging.info("Read file success")
            return dataframe

    except FileNotFoundError:
            print("No such file or directory")
            logging.error()
    except AssertionError:
            print("Input path is not string format")
            logging.error()

def perform_eda(dataframe: pd.DataFrame) -> None:
    '''
    Perform EDA on raw dataframe.
    List of analysis are 
        1.) Number of missing value in each columns
        2.) 
    Then save results and figures to images folder.
    Args:
            dataframe (pd.DataFrame): Pandas DataFrame with raw data
            target_col (str) : Column name of target prediction

    Outputs:
            None
    '''


    pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__ == '__main__':
        raw_data = import_data("./data/bank_data.csv")
        print(settings.features_select['numerical'])
        