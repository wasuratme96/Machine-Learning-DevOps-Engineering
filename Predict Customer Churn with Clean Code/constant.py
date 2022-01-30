# Path In-Out 
path = dict(
    RAW_DATA_PATH = './data',
    DATA_PROFILE_PATH = './data/data profile',
    IMG_EDA_PATH = './images/eda',
    IMG_RESULT_PATH = './images/results',
    MODEL_RESULT_PATH = './models',
)

# Plotting Parameters
eda_plot = dict(
    width = 20,
    height = 10,
    annot = False,
    cmap = 'Dark2_r',
    linewidths = 2,   
)

result_plot = dict(
    width = 15,
    height = 8,
    alpha = 0.8

)

# Data Labelling Constant
label = dict(
    class_0 = 'Existing Customer',
    class_1 = 'Attrited Customer'
)

# Machine Learning Constant
feature_select = [
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
    'Card_Category_Churn'
]

test_portions = 0.3

seed = dict(
    seed_number  = 112233
)

rf_parms = dict(
    n_estimators = [200, 500],
    max_features = ['auto', 'sqrt'],
    max_depth = [4,5,100],
    criterion = ['gini', 'entropy'],
    cv = 5
)

lr_parms = dict(
    max_iter = 1000
)