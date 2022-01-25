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
from numbers import Number

# Data Manipulations Library
import pandas as pd
import numpy as np

# Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import asec; sns.set()

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
                countplot_list: list = [],
                distplot_list : list = [],
                normcountplot_list: list = [],
                img_output_path: str = './images/eda',
                data_profile_path: str = './data/data profile') -> None:
    '''
    Perform EDA on raw dataframe with list list of items

    1.) Number of missing value in each columns

    2.) Basic Descriptive Statistic

    3.) Histogram of amount in each predict class

    4.) Histogram of selected columns

    5.) Normalized Barplot of selected columns

    6.) Heatmap of all columns in dataset

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
    except KeyError:
            print("Given target prediction columns is not available in dataframe")
            logging.error("ERROR : Target prediction columns is not available in dataframe")

    try:
            # Ensure non-empty file
            assert dataframe.shape[0] > 0

            # Missing value properties
            pd.DataFrame(dataframe.isnull().sum(), columns = ['Null Value']).to_csv(f'{data_profile_path}/missing_value.csv')
            # Basic descriptive statistics
            dataframe.describe().to_csv(f'{data_profile_path}/basic_stat.csv')

            # Heat Map Plot
            plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
            sns.heatmap(dataframe.corr(), 
                        annot= settings.eda_plot['annot'], 
                        cmap= settings.eda_plot['cmap'], 
                        linewidths = settings.eda_plot['linewidths'])
            plt.savefig(f'{img_output_path}/Correlation_heatmap', bbox_inches = 'tight')

    except AssertionError:
            print("Dataframe is empty !")
            logging.error("ERROR : Dataframe is empty ")

    try:
            # Check count-plot columns list
            assert len(countplot_list) != 0

            # Count Plot
            countplot_list.append(target_col)
            for col_name in countplot_list:
                    plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
                    sns.countplot(x = dataframe[col_name])
                    plt.savefig(f'{img_output_path}/{col_name}_countplot', bbox_inches = 'tight')

    except AssertionError:
            print("No features given for Count Plot")
            logging.info("INFO : No features given for Count Plot")

    try:
           assert len(normcountplot_list) > 0 

           # Normalized Count Plot
           for col_name in normcountplot_list:
                   plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
                   dataframe[col_name].value_counts('normalize').plot.bar()
                   plt.savefig(f'{img_output_path}/{col_name}_normcountplot', bbox_inches = 'tight')

    except AssertionError:
            print("No features given for Normalized Count Plot")
            logging.info("INFO : No features given for Normalized Count Plot")


    try:    
            assert len(distplot_list) > 0

            # Distribution Plot
            for col_name in distplot_list:
                    plt.figure(figsize = (settings.eda_plot['width'], settings.eda_plot['height']))
                    sns.displot(x = dataframe[col_name])
                    plt.savefig(f'{img_output_path}/{col_name}_distplot', bbox_inches = 'tight')
    except AssertionError:
            print("No features given for Distribution Plot")
            logging.info("INFO : No features given for Distribution Plot")    

def encoder_helper(dataframe: pd.DataFrame, 
                   category_lst: list = []) -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    propotion of "Churn" for each category.

    Args:
    ----------
            dataframe (pd.DataFrame): pandas dataframe
            category_lst: list of columns that contain categorical features

    Outputs:
    ----------
            dataframe (pd.DataFrame): pandas dataframe with new columns from encoder
    '''
    try :
            # Check if all given columns is available in dataframe
            assert len(set(category_lst).difference(set(dataframe.columns))) == 0
            
            for encode_col in category_lst:
                    encode_group = dataframe.groupby(encode_col).mean()['Churn']
                    dataframe[encode_col + "_Churn"] = dataframe[encode_col].apply(lambda x : dict(encode_group)[x])
                    
            return dataframe

    except AssertionError:
            print("Given target encode columns is out of dataframe columns")
            logging.error("ERROR : Given target encode columns is out of dataframe columns")

def perform_feature_engineering(dataframe: pd.DataFrame, 
                                features_list:list,
                                test_ratio: Number = None) -> pd.DataFrame:
    '''
    Seperate feature for model development and target prediction.
    Split train - test data into given percentage

    Args:
    ----------
              dataframe (pd.DataFrame): pandas dataframe
              features_list (list): List of selected features name of model development
              test_ratio 

    Outputs:
    ----------
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
            assert len(set(features_list).difference(set(dataframe.columns))) == 0
            
            X = dataframe[features_list]
            y = dataframe['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_ratio, random_state=settings.seed['seed_number'])
            return X_train, X_test, y_train, y_test

    except AssertionError:
            print("Given selected columns is out of dataframe columns")
            logging.error("ERROR : Given selected columns is out of dataframe columns")

    except ValueError:
            print(f"""test_ratio should be positive integer with smaller than number of samples {len(dataframe)} or 
            a float in (0, 1) range""")
            logging.error("ERROR : Given test_ratio is out of range or negative value")

                
def classification_report_image(y_train: np.ndarray,
                                y_test: np.ndarray,
                                y_train_preds_lr: np.ndarray,
                                y_train_preds_rf: np.ndarray,
                                y_test_preds_lr: np.ndarray,
                                y_test_preds_rf: np.ndarray,
                                img_output_path: str) -> None:
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder

    Args:
    ----------
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    Outputs:
    ----------
             None
    '''
    results_dict = {
            'Logistic Regresion': [y_train_preds_lr, y_test_preds_lr],
            'Random Forest' : [y_train_preds_rf, y_test_preds_rf]
    }

    for model_name, predict_value in results_dict.items():
        plt.rc('figure', figsize=(5, 5))
        # Train Results
        plt.text(0.01, 1.25, str(f'{model_name} Train'), 
                 {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, predict_value[0])), 
                 {'fontsize': 10}, fontproperties = 'monospace')

        # Test Results
        plt.text(0.01, 0.6, str(f'{model_name} Test'), 
                 {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, predict_value[1])), 
                 {'fontsize': 10}, fontproperties = 'monospace')
        plt.axis('off');

        plt.savefig(f"{img_output_path}/{model_name} Classification Report", bbox_inches = 'tight')
    pass


def feature_importance_plot(model, 
                            X_data: np.ndarray, 
                            img_output_path: str) -> None:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            img_output_path: path to store the figure
    output:
             None
    '''
    # SHAPE Feature Importance
    explainer = shap.TreeExplainer(model.best_estimator_) #Calculate feature importance
    shap_values = explainer.shap_values(X_data)
    
    # Create plot
    shap_fig = plt.gcf()
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    shap_fig.savefig(f'{img_output_path}/SHAP TreeExplainer', bbox_inches = 'tight')
    
    # Built-In Random Forest Feature Importance
    importances = model.best_estimator_.feature_importances_  #Calculate feature importance
    indices = np.argsort(importances)[::-1] #Sort importance score in descending order
    names = [X_data.columns[i] for i in indices] #Sort feature name accord to score
    
    # Create plot
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(f'{img_output_path}/RandomForest Importance Score', bbox_inches = 'tight')

    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    Traina and store model results: images + scores, and store models.
    Args:
    ----------
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    Outputs:
    ----------
              None
    '''
    pass

if __name__ == '__main__':
        # Read in data
        raw_data = import_data("./data/bank_data.csv")
        
        # Perform exploratory dat analysis 
        countplot_list = ['Education_Level', 'Income_Category']
        distplot_list = ['Total_Trans_Ct', 'Total_Trans_Amt']
        normcountplot_list = ['Marital_Status', 'Card_Category']

        perform_eda(raw_data, 
                    'Attrition_Flag', 
                    countplot_list, 
                    distplot_list,
                    normcountplot_list)

        # Encoding all categorical features
        all_category_features = raw_data.select_dtypes('object').columns

        ## Exclude target prediction columns
        category_features = [colname != 'Attrition_Flag' for colname in all_category_features]
        encoded_data = encoder_helper(raw_data, category_features)

        # Feature engineering
        selected_features = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

        X_train, X_test, y_train, y_test = perform_feature_engineering(encoded_data, selected_features, 0.3)
        
        
        