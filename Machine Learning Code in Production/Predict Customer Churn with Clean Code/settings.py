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

)

features_select = dict(
    categorical = [
                    'Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category'
                    ],
    numerical = [
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
            'Avg_Utilization_Ratio'
            ]
)
