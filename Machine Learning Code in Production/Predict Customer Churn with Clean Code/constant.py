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