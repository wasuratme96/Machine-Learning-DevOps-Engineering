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

label = dict(
    class_0 = 'Existing Customer',
    class_1 = 'Attrited Customer'
)

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