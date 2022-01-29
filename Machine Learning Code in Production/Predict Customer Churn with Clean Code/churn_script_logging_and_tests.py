'''
To test and log the passes or error on each function in churn_library.py
All information will be stored in ./logs/test_churn_library.log

Usage : ./pytest churn_script_logging_and_tests.py
Authors : Wasurat Soontronchai <wasurat_me96@outlook.com>
Date : 27 Jan 2022
'''
import os
import logging

import pytest
import joblib

import churn_library as cl

logging.basicConfig(
    filename='./logs/test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
test_logging = logging.getLogger("UNIT TEST RUN")

def test_import(file_path):
	'''
	test import_data functions
	'''
	try:
		assert isinstance(file_path, str)
		df = cl.import_data(file_path)
		test_logging.info("Testing import_data: SUCCESS")
	except AssertionError as err:
		test_logging.error("Testing import_data: given file path is not string")
		raise err

	except FileNotFoundError as err:
		test_logging.error("Testing import_data: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		test_logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

def test_eda(perform_eda):
	'''
	test perform eda function
	'''


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	test_import(file_path = './data/bank_data.csv')
	








