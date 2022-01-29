'''
To test and log the passes or error on each function in churn_library.py
All information will be stored in ./logs/test_churn_library.log

Usage : ./pytest churn_script_logging_and_tests.py
Authors : Wasurat Soontronchai <wasurat_me96@outlook.com>
Date : 27 Jan 2022
'''
import os
import logging
from typing import Dict
import pandas as pd
import pytest
import joblib

import churn_library as cl

#logging.basicConfig(
#    filename='./logs/test_results.log',
#    level=logging.INFO,
#    filemode='w',
#    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#)
#test_logging = logging.getLogger("UNIT TEST RUN")

# Logger Settings
test_logging = logging.getLogger("UNIT TEST RUN")
test_logging.setLevel(logging.INFO)

fh = logging.FileHandler('./logs/test_results.log')
fh.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)

test_logging.addHandler(fh)

def test_import(import_data, file_path):
	'''
	Test import_data functions on given file_path
	Args:
		- import_data (Pytho function):
		- file_path (str) :
	Output:
		None
	'''
	try:
		assert isinstance(file_path, str)
		df = import_data(file_path)
		test_logging.info("Testing import_data: SUCCESS")
		try :
			assert df.shape[0] > 0
			assert df.shape[1] > 0
		except AssertionError as err:
			test_logging.error('''Testing import_data: 
								   The file doesn't appear to have rows and columns''')

	except AssertionError as err:
		test_logging.error("Testing import_data: given file path is not string")
	except FileNotFoundError as err:
		test_logging.error("Testing import_data: The file wasn't found")
	except pd.errors.EmptyDataError as err:
		test_logging.error("Testing import_data: The file is empty")

def test_eda(perform_eda: function,
			eda_param_dict: Dict,
			import_data: function,
			file_path: str ) -> None:
	'''
	Test perform eda function
	'''
	raw_data = import_data(file_path)

	perform_eda(raw_data,
				eda_param_dict['target_col'],
				eda_param_dict['countplot_columns'],
				eda_param_dict['distplot_columns'],
				eda_param_dict['normcountplot_columns']
	)



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
	# Test import_data function
	test_import(cl.import_data, './data/bank_data.csv')
	test_import(cl.import_data, './data/not_exist_path.csv')
	test_import(cl.import_data, './data/test_empty_file.csv')

	# Test perform_eda function
	eda_test_1 = {
		'target_col' : 'Attrition_Flag',
		'countplot_columns' : ['Education_Level', 'Income_Category'],
		'distplot_columns' : ['Total_Trans_Ct', 'Total_Trans_Amt'],
		'normcountplot_columns' : ['Marital_Status', 'Card_Category']
	}
	eda_test_2 = {
		'target_col' : 'Attrition_Flag',
		'countplot_columns' : ['Education_Level', 'Income_Category'],
		'distplot_columns' : ['Total_Trans_Ct', 'Total_Trans_Amt'],
		'normcountplot_columns' : ['Marital_Status', 'Card_Category']
	}
	test_eda(cl.perform_eda, eda_test_1,cl.import_data, './data/bank_data.csv')
	test_eda(cl.perform_eda, eda_test_2,cl.import_data, './data/bank_data.csv')

	








