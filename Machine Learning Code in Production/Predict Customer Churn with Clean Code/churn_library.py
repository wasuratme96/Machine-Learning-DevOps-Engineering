'''
Use to predict churn of bank customer with binary classification problem.
Selected model from Sklearn are LogisticRegression and RandomForestClassifier.
Exploratory data analysis and model tuning with Gridsearch method is included.

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

def import_data(file_path: str) -> pd.DataFrame:
    '''
    Returns dataframe for the csv given from file path.
    And specific index_col = 0 to remove unwanted column.

    Args:
    ----------
            file_path (str): A path to the csv file

    Outsputs:
    ----------
            dataframe (pd.DataFrame): pandas dataframe
    '''
    try:
            assert isinstance(file_path, str)

            dataframe = pd.read_csv(file_path, index_col = 0)
            logging.info("Read file success")
            return dataframe

    except FileNotFoundError:
            print("No such file or directory from given path")
            logging.error("ERROR : No such file or directory from given path")
    except AssertionError:
            print("Input path is inccorrect format")
            logging.error("ERROR : Input path is inccorrect format")

def perform_eda(dataframe: pd.DataFrame,
                target_col: str,
                countplot_list: list,
                distplot_list: list,
                normcountplot_list: list) -> None:
    '''
    Perform EDA on raw dataframe with list list of items

    1.) Number of missing value in each columns :

    2.) Basic Descriptive Statistic :

    3.) Histogram of amount in each predict class :

    4.) Histogram of selected columns : 

    5.) Normalized Barplot of selected columns :

    6.) Heatmap of all columns in dataset :

    Then save results and figures to .images/eda folder.

    Args:
    ----------
            dataframe (pd.DataFrame): Pandas DataFrame with raw data
            target_col (str) : Column name of target prediction

    Outputs:
    ----------
            None
    '''
    try:
            # Ensure the binary classification problem
            assert len(dataframe[target_col].unique()) == 2 

            # Change target column to numerical value 
            dataframe['Churn'] = np.where(dataframe[target_col] == settings.label['class_0'], 0, 1)

    except AssertionError:
            print("Target prediction is not binary problem")
            logging.error("ERROR : Target prediction is not binary problem")
    
    data_profile_path = './data/data profile'
    img_path = './images/eda'

    # Missing Value Check
    pd.DataFrame(dataframe.isnull().sum(), columns = ['Null Value']).to_csv(f'{data_profile_path}/missing_value.csv')
    # Basic Descriptive Statistics
    dataframe.describe().to_csv(f'{data_profile_path}/basic_stat.csv')
    
    # Count Plot
    countplot_list.append(target_col)
    for col_name in countplot_list:
            plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
            sns.countplot(x = dataframe[col_name])
            plt.savefig(f'{img_path}/{col_name}_countplot')

    # Normalized Count Plot
    for col_name in normcountplot_list:
            plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
            dataframe['Marital_Status'].value_counts('normalize').plot.bar()
            plt.savefig(f'{img_path}/{col_name}_normcountplot')

    # Distribution Plot
    for col_name in distplot_list:
            plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
            sns.displot(x = dataframe[col_name])
            plt.savefig(f'{img_path}/{col_name}_distplot')

    # Heat Map
    plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
    sns.heatmap(dataframe.corr(), 
                annot= settings.eda_plot['annot'], 
                cmap= settings.eda_plot['cmap'], 
                linewidths = settings.eda_plot['linewidths'])
    plt.savefig(f'{img_path}/Correlation_heatmap')

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
        # Read in data
        raw_data = import_data("./data/bank_data.csv")
        
        # Perform exploratory dat analysis 
        countplot_list = ['Education_Level', 'Income_Category']
        distplot_list = ['Total_Trans_Ct', 'Total_Trans_Amt']
        normcountplot_list = ['Marital_Status']
        perform_eda(raw_data, 
                    'Attrition_Flag', 
                    countplot_list, 
                    distplot_list,
                    normcountplot_list)
        
        