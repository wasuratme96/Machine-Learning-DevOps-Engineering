'''
To test and log the passes or error on each function in churn_library.py
All information will be stored in ./logs/test_results.log

Usage : ./python churn_script_logging_and_tests.py
Authors : Wasurat Soontronchai <wasurat_me96@outlook.com>
Date : 29 Jan 2022
'''
import os
import logging
from typing import Dict
from typing import Callable
from numbers import Number
import joblib

import pandas as pd
import numpy as np

import churn_library as cl
import constant

# Logger Settings
test_logging = logging.getLogger("UNIT TEST RUN")
test_logging.setLevel(logging.INFO)

fh = logging.FileHandler('./logs/test_results.log')
fh.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)

test_logging.addHandler(fh)


def test_import(import_data: Callable,
                file_path: str) -> None:
    '''
    Test import_data functions on given file_path
    Args:
            - import_data (function):
            - file_path (str):
    Output:
            None
    '''
    try:
        assert isinstance(file_path, str)
        dataframe = import_data(file_path)
        test_logging.info("Testing import_data: SUCCESS")
        try:
            assert dataframe.shape[0] > 0
            assert dataframe.shape[1] > 0
        except AssertionError as err:
            test_logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")

    except AssertionError as err:
        test_logging.error(
            "Testing import_data: given file path is not string")
        raise err
    except FileNotFoundError as err:
        test_logging.error("Testing import_data: The file wasn't found")
        raise err
    except pd.errors.EmptyDataError as err:
        test_logging.error("Testing import_data: The file is empty")
        raise err


def test_eda(perform_eda: Callable,
             eda_param_dict: Dict,
             import_data: Callable,
             file_path: str) -> None:
    '''
    Test perform eda function on given attribute dict.
    Expected result are 2 data profile in .csv format and
    images of plotting in .png base on group of plotting.

    Args:
            - perform_eda (Callable):
            - eda_param_dict (Dict):
            - import_data (Callable):
            - file_path (str):
    Outputs:
            None
    '''
    # Action for function
    raw_data = import_data(file_path)

    # Test Functions
    perform_eda(raw_data,
                eda_param_dict['target_columns'],
                eda_param_dict['countplot_columns'],
                eda_param_dict['distplot_columns'],
                eda_param_dict['normcountplot_columns'])

    # Data profile creation checking
    for file_name in ["missing_value", "basic_stat"]:
        try:
            assert os.path.exists("./data/data profile/%s.csv" %file_name)
            test_logging.info(
                "Testing perform_eda: %s.csv creation SUCCESS", file_name)
        except AssertionError as err:
            test_logging.info(
                "Testing perform_eda: %s.csv creation FAIL", file_name)
            raise err

    # Heatmap images creation checking
    try:
        assert os.path.exists("./images/eda/Correlation_heatmap.png")
        test_logging.info(
            "Testing perform_eda: Correlation_heatmap creation SUCCESS")
    except AssertionError as err:
        test_logging.error(
            "Testing perform_eda: Correlation_heatmap creation FAIL")
        raise err

    # Plotting by type of plot checking
    for key, value_list in eda_param_dict.items():
        if key != 'target_columns':
            plot_type = key.replace('_columns', '')
            for plot_name in value_list:
                try:
                    assert os.path.exists(
                        "./images/eda/%s_%s.png" %
                        (plot_name, plot_type))
                    test_logging.info(
                        "Testing perform_eda: %s_%s.png creation SUCCESS" %
                        (plot_name, plot_type))
                except AssertionError as err:
                    test_logging.error(
                        "Testing perform_eda: %s_%s.png creation FAIL" %
                        (plot_name, plot_type))
                    raise err


def test_encoder_helper(encoder_helper: Callable,
                        import_data: Callable,
                        file_path: str) -> None:
    '''
    Test encoder_helper on all given 'object' columns list.
    Checking exising of given columns name with suffix _Churn and
    datatype of that column is not object.

    Args:
            - encoder_helper (Callable):
            - import_data (Callable):
            - file_path (str):
    Outputs:
            None
    '''
    # Action for function
    raw_data = import_data(file_path)
    raw_data['Churn'] = np.where(
        raw_data['Attrition_Flag'] == constant.label['class_0'], 0, 1)
    all_category_features = raw_data.select_dtypes('object').columns
    category_features = [
        colname for colname in all_category_features if colname != 'Attrition_Flag']

    # Test Function
    encoded_data = encoder_helper(raw_data, category_features)

    for col_name in category_features:
        encoded_col = col_name + '_Churn'
        try:
            assert encoded_col in encoded_data.columns
            assert encoded_data[encoded_col].dtypes != 'O'
            test_logging.info(
                "Testing encoder_helper: %s column encode SUCCESS",
                encoded_col)

        except AssertionError as err:
            test_logging.error(
                "Testing encoder_helper: %s column encode FAIL", encoded_col)
            raise err


def test_perform_feature_engineering(perform_feature_engineering: Callable,
                                     selected_features: list,
                                     test_portions: Number,
                                     import_data: Callable,
                                     file_path: str) -> None:
    '''
    Test perform_feature_engineering on given all preprocess data.
    Expected output is features and target have same length and
    proportion of train/test number match with test_portions

    Args:
            - perform_feature_engineering (Callable):
            - import_data (Callable):
            - test_portions (Number):
            - file_path (str):
    Outputs:
            None
    '''
    # Action for function
    raw_data = import_data(file_path)
    raw_data['Churn'] = np.where(
        raw_data['Attrition_Flag'] == constant.label['class_0'], 0, 1)
    all_category_features = raw_data.select_dtypes('object').columns
    category_features = [
        colname for colname in all_category_features if colname != 'Attrition_Flag']
    encoded_data = cl.encoder_helper(raw_data, category_features)

    # Test function
    xtrain, xtest, ytrain, ytest = perform_feature_engineering(
        encoded_data, selected_features, test_portions)
    try:
        assert len(xtrain) == len(ytrain)
        assert len(xtest) == len(ytest)
        assert round(len(xtest) / len(encoded_data), 2) == test_portions
        test_logging.info(
            "Testing perform_feature_engineering: Train and Test split SUCCESS")
    except AssertionError as err:
        test_logging.info(
			            "Testing perform_feature_engineering: Train and Test split FAIL")
        raise err


def test_train_models(train_models: Callable,
                      import_data: Callable,
                      file_path: str) -> None:
    '''
    Test train_models functions by checking on below expected items.
    1.) Model file creation and test prediction
    2.) Classification report creation
    3.) Feature importance creation
    4.) ROC Curve plot creation

    Args:
            - train_models (Callable):
            - import_data (Callable):
            - file_path (str):
    Outputs:
            None
    '''
    # Action for function
    raw_data = import_data(file_path)
    raw_data['Churn'] = np.where(
        raw_data['Attrition_Flag'] == constant.label['class_0'], 0, 1)
    all_category_features = raw_data.select_dtypes('object').columns
    category_features = [
        colname for colname in all_category_features if colname != 'Attrition_Flag']
    encoded_data = cl.encoder_helper(raw_data, category_features)
    xtrain, xtest, ytrain, ytest = cl.perform_feature_engineering(
        encoded_data, constant.feature_select, constant.test_portions)

    # Test function
    train_models(xtrain, xtest, ytrain, ytest)

    # Model file checking
    for model_type in ['logistic', 'rfc']:
        try:
            assert os.path.exists("./models/%s_model.pkl" %model_type)
            test_logging.info(
                "Testing train_models: %s model creation SUCCESS", model_type)

            try:
                loaded_model = joblib.load(
                    "./models/%s_model.pkl", model_type)
                assert len(loaded_model.predict(xtest)) == len(ytest)
                test_logging.info(
                    "Testing train_models: %s model prediction SUCCESS",
                    model_type)
            except AssertionError as err:
                test_logging.error(
                    "Testing train_models: %s model prediction FAIL",
                    model_type)

        except AssertionError as err:
            test_logging.error(
                "Testing train_models: %s model creation FAIL" , model_type)
            raise err

    # Classification report checking
    for model_name in ['Logistic Regresion', 'Random Forest']:
        try:
            assert os.path.exists(
                "./images/results/%s Classification Report.png" %model_name)
            test_logging.info(
                "Testing train_models: %s Classification Report creation SUCCESS",
                model_name)
        except AssertionError as err:
            test_logging.info(
                "Testing train_models: %s Classification Report creation FAIL",
                model_name)

    # Feature importance checking
    for importance_type in ['SHAP TreeExplainer', 'TreeBase Impurity Score']:
        try:
            assert os.path.exists(
                "./images/results/%s.png" %importance_type)
            test_logging.info(
                "Testing train_models: %s creation SUCCESS", importance_type)
        except AssertionError as err:
            test_logging.info(
                "Testing train_models: %s creation FAIL", importance_type)

    # ROC curve plot checking
    for data_set in ['Train Data', 'Test Data']:
        try:
            assert os.path.exists(
                "./images/results/%s ROC Curve.png" %data_set)
            test_logging.info(
                "Testing train_models: %s ROC Curve creation SUCCESS",
                data_set)
        except AssertionError as err:
            test_logging.info(
                "Testing train_models: %s ROC Curve creation  FAIL", data_set)


if __name__ == "__main__":
    # Test import_data function
    test_import(cl.import_data,
                f"{constant.path['RAW_DATA_PATH']}/bank_data.csv")

    # Test perform_eda function
    eda_test = {
        'target_columns': 'Attrition_Flag',
        'countplot_columns': ['Education_Level', 'Income_Category'],
        'distplot_columns': ['Total_Trans_Ct', 'Total_Trans_Amt'],
        'normcountplot_columns': ['Marital_Status', 'Card_Category']
    }
    test_eda(cl.perform_eda, eda_test, cl.import_data,
             f"{constant.path['RAW_DATA_PATH']}/bank_data.csv")

    # Test encoder_helper function
    test_encoder_helper(cl.encoder_helper, cl.import_data,
                        f"{constant.path['RAW_DATA_PATH']}/bank_data.csv")

    # Test perform_feature_engineering
    select_features = constant.feature_select
    TEST_RATIO = constant.test_portions
    test_perform_feature_engineering(
        cl.perform_feature_engineering,
        select_features,
        TEST_RATIO,
        cl.import_data,
        f"{constant.path['RAW_DATA_PATH']}/bank_data.csv")

    # Test train_models
    test_train_models(cl.train_models,
                      cl.import_data,
                      f"{constant.path['RAW_DATA_PATH']}/bank_data.csv")
