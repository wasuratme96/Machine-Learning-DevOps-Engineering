'''
Use to predict churn of bank customer with binary classification problem.
Selected model from Sklearn are LogisticRegression and RandomForestClassifier.
Exploratory data analysis and model tuning with Gridsearch method is included.

Usage : ./churn_library.py
Authors : Wasurat Soontronchai <wasurat_me96@outlook.com>
Date : 27 Jan 2022
'''

# Standard Library
from typing import TypeVar
from typing import Dict
from numbers import Number
import logging

# Machine Learning Related Libary
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib

# Data Manipulations Library
import pandas as pd
import numpy as np

# Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Settings and Constant
import constant
Predictor = TypeVar('Predictor')

# Logger Settings
run_logging = logging.getLogger("SCRIPT RUN LOG")
run_logging.setLevel(logging.INFO)

fh = logging.FileHandler('./logs/run_results.log')
fh.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)

run_logging.addHandler(fh)

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

        run_logging.info("File path : %s", str(file_path))
        assert isinstance(file_path, str)

        dataframe = pd.read_csv(file_path, index_col=0)

        run_logging.info("Read file success")
        run_logging.info("Dataframe have %s rows", dataframe.shape[0])
        run_logging.info("Dataframe have %s columns", dataframe.shape[1])
        return dataframe

    except FileNotFoundError as err:
        print("No such file or directory from given path")
        run_logging.error("ERROR: No such file/ directory from given path")
        raise err
    except pd.errors.EmptyDataError as err:
        print("Input file is empty")
        run_logging.error("ERROR: Input file is empty")
        raise err
    except AssertionError as err:
        print("Input path is inccorrect format")
        run_logging.error("ERROR: Input path is inccorrect format")
        raise err

def perform_eda(dataframe: pd.DataFrame,
                target_col: str,
                countplot_list: list,
                distplot_list: list,
                normcountplot_list: list) -> None:
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
        dataframe['Churn'] = np.where(
            dataframe[target_col] == constant.label['class_0'], 0, 1)

    except AssertionError:
        print("Target prediction is not binary problem")
        run_logging.error("ERROR : Target prediction is not binary problem")
    except KeyError:
        print("Given target prediction columns is not available in dataframe")
        run_logging.error(
            "ERROR : Target prediction columns is not available in dataframe")

    try:
        # Ensure non-empty file
        assert dataframe.shape[0] > 0

        # Missing value properties
        pd.DataFrame(dataframe.isnull().sum(), columns=['Null Value'])\
                .to_csv(f"{constant.path['DATA_PROFILE_PATH']}/missing_value.csv")
        # Basic descriptive statistics
        dataframe.describe()\
                .to_csv(f"{constant.path['DATA_PROFILE_PATH']}/basic_stat.csv")

        # Heat Map Plot
        plt.figure(
            figsize=(
                constant.eda_plot['width'],
                constant.eda_plot['height']))
        sns.heatmap(dataframe.corr(),
                    annot=constant.eda_plot['annot'],
                    cmap=constant.eda_plot['cmap'],
                    linewidths=constant.eda_plot['linewidths'])
        plt.savefig(
            f"{constant.path['IMG_EDA_PATH']}/Correlation_heatmap",
            bbox_inches='tight')
        plt.close()

    except AssertionError:
        print("Dataframe is empty !")
        run_logging.error("ERROR : Dataframe is empty ")

    try:
        # Check count-plot columns list
        assert len(countplot_list) != 0

        # Count Plot
        countplot_list.append(target_col)
        for col_name in countplot_list:
            plt.figure(
                figsize=(
                    constant.eda_plot['width'],
                    constant.eda_plot['height']))
            sns.countplot(x=dataframe[col_name])
            plt.savefig(
                f"{constant.path['IMG_EDA_PATH']}/{col_name}_countplot",
                bbox_inches='tight')
            plt.close()

    except AssertionError:
        print("No features given for Count Plot")
        run_logging.info("INFO : No features given for Count Plot")

    try:
        assert len(normcountplot_list) > 0

        # Normalized Count Plot
        for col_name in normcountplot_list:
            plt.figure(
                figsize=(
                    constant.eda_plot['width'],
                    constant.eda_plot['height']))
            dataframe[col_name].value_counts('normalize').plot.bar()
            plt.savefig(
                f"{constant.path['IMG_EDA_PATH']}/{col_name}_normcountplot",
                bbox_inches='tight')
            plt.close()

    except AssertionError:
        print("No features given for Normalized Count Plot")
        run_logging.info("INFO : No features given for Normalized Count Plot")

    try:
        assert len(distplot_list) > 0

        # Distribution Plot
        for col_name in distplot_list:
            plt.figure(
                figsize=(
                    constant.eda_plot['width'],
                    constant.eda_plot['height']))
            sns.displot(x=dataframe[col_name])
            plt.savefig(
                f"{constant.path['IMG_EDA_PATH']}/{col_name}_distplot",
                bbox_inches='tight')
            plt.close()

    except AssertionError:
        print("No features given for Distribution Plot")
        run_logging.info("INFO : No features given for Distribution Plot")

def encoder_helper(dataframe: pd.DataFrame,
                   category_lst: list) -> pd.DataFrame:
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
    try:
        # Check if all given columns is available in dataframe
        assert len(set(category_lst).difference(set(dataframe.columns))) == 0

        for col_name in category_lst:
            col_group = dataframe.groupby(col_name).mean()['Churn']
            dataframe[col_name + "_Churn"] = dataframe[col_name].apply(lambda x: dict(col_group)[x])

        return dataframe
    except AssertionError:
        print("Given target encode columns is out of dataframe columns")
        run_logging.error(
            "ERROR : Given target encode columns is out of dataframe columns")


def perform_feature_engineering(dataframe: pd.DataFrame,
                                features_list: list,
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
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        assert len(set(features_list).difference(set(dataframe.columns))) == 0

        features = dataframe[features_list]
        target = dataframe['Churn']
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_ratio, random_state=constant.seed['seed_number'])
        return x_train, x_test, y_train, y_test

    except AssertionError:
        print("Given selected columns is out of dataframe columns")
        run_logging.error(
            "ERROR : Given selected columns is out of dataframe columns")

    except ValueError:
        print(
            f"""test_ratio should be positive int with smaller than number
            of data {len(dataframe)} ora float in (0, 1) range""")
        run_logging.error(
            "ERROR : Given test_ratio is out of range or negative value")


def classification_report_image(y_actual_dict: Dict,
                                y_train_preds_lr: np.ndarray,
                                y_train_preds_rf: np.ndarray,
                                y_test_preds_lr: np.ndarray,
                                y_test_preds_rf: np.ndarray) -> None:
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
    y_train = y_actual_dict['train']
    y_test = y_actual_dict['test']
    results_dict = {
        'Logistic Regresion': [y_train_preds_lr, y_test_preds_lr],
        'Random Forest': [y_train_preds_rf, y_test_preds_rf]
    }

    for model_name, predict_value in results_dict.items():
        plt.figure('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str(f'{model_name} Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_train, predict_value[0])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model_name} Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_test, predict_value[1])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            f"{constant.path['IMG_RESULT_PATH']}/{model_name} Classification Report",
            bbox_inches='tight')
        plt.close()

def feature_importance_plot(model: Predictor,
                            x_data: np.ndarray) -> None:
    '''
    Creates and stores the feature importances in IMG_OUTPUT_PATH.
    SHAP value and feature importance built-in random forest are selected.

    Args:
    ----------
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
    Outputs:
    ----------
             None
    '''
    # SHAPE Feature Importance
    # Calculate feature importance
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_data)

    # Create plot
    plt.figure(figsize=(20, 5))
    shap_fig = plt.gcf()
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    shap_fig.savefig(
        f"{constant.path['IMG_RESULT_PATH']}/SHAP TreeExplainer",
        bbox_inches='tight')
    plt.close()

    # Built-In Random Forest Feature Importance
    # Calculate feature importance
    importances = model.best_estimator_.feature_importances_
    # Sort importance score in descending order
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i]
             for i in indices]  # Sort feature name accord to score

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(
        f"{constant.path['IMG_RESULT_PATH']}/RandomForest Importance Score",
        bbox_inches='tight')
    plt.close()

def train_models(x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray) -> None:
    '''
    Train and store model results: images + scores, and store models.
    Selected model are LogistricRegression and RandomForest.
    Hyperparameter tuning have been done for RandomForest

    Args:
    ----------
              x_train (np.ndarray): X training data
              x_test (np.ndarray): X testing data
              y_train (np.ndarray): y training data
              y_test (np.ndarray): y testing data
    Outputs:
    ----------
              None
    '''

    # Random Forest
    rfc = RandomForestClassifier(random_state=constant.seed['seed_number'])
    param_grid = {
        'n_estimators': constant.rf_parms['n_estimators'],
        'max_features': constant.rf_parms['max_features'],
        'max_depth': constant.rf_parms['max_depth'],
        'criterion': constant.rf_parms['criterion']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=constant.rf_parms['cv'])
    cv_rfc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # Logistic Regression
    lrc = LogisticRegression(max_iter=constant.lr_parms['max_iter'])
    lrc.fit(x_train, y_train)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Save best model into model_path
    joblib.dump(cv_rfc.best_estimator_, f"{constant.path['MODEL_RESULT_PATH']}/rfc_model.pkl")
    joblib.dump(lrc, f"{constant.path['MODEL_RESULT_PATH']}/logistic_model.pkl")

    # Performance plot
    # Classification Report
    y_true_dict = {"train" : y_train, "test" : y_test}
    classification_report_image(y_true_dict,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)
    # Features Importance Plot
    feature_importance_plot(cv_rfc, x_test)

    # ROC Curve for train and test data
    data_set = {
        'Train Data': [
            x_train, y_train],
        'Test Data': [
            x_test, y_test]}
    for data, data_value in data_set.items():
        plt.figure(
            figsize=(
                constant.result_plot['width'],
                constant.result_plot['height']))
        fig_plot = plt.gca()
        plot_roc_curve(
            lrc,
            data_value[0],
            data_value[1],
            ax=fig_plot,
            alpha=constant.result_plot['alpha'])
        plot_roc_curve(
            cv_rfc.best_estimator_,
            data_value[0],
            data_value[1],
            ax=fig_plot,
            alpha=constant.result_plot['alpha'])
        plt.title(data)
        plt.savefig(f"{constant.path['IMG_RESULT_PATH']}/{data} ROC Curve")
        plt.close()

if __name__ == '__main__':
    # Read in data
    raw_data = import_data(f"{constant.path['RAW_DATA_PATH']}/bank_data.csv")

    # Perform exploratory dat analysis
    countplot_columns = ['Education_Level', 'Income_Category']
    distplot_columns = ['Total_Trans_Ct', 'Total_Trans_Amt']
    normcountplot_columns = ['Marital_Status', 'Card_Category']

    perform_eda(raw_data,
                'Attrition_Flag',
                countplot_columns,
                distplot_columns,
                normcountplot_columns)

    # Encoding all categorical features
    all_category_features = raw_data.select_dtypes('object').columns

    # Exclude target prediction columns
    category_features = [
        colname for colname in all_category_features if colname != 'Attrition_Flag']
    encoded_data = encoder_helper(raw_data, category_features)

    # Feature engineering
    selected_features = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    xtrain, xtest, ytrain, ytest = perform_feature_engineering(
        encoded_data, selected_features, 0.3)
    train_models(
        xtrain,
        xtest,
        ytrain,
        ytest)
