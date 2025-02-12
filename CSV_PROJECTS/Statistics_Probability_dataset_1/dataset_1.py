
#               ---- IMPORT_REQUIRED_LIBRARIES ----
import pandas as pd
import numpy as np
from scipy.stats import norm 
from scipy.stats import ttest_ind
import statsmodels.api as sm
import warnings

# Load the Dataset
dataset = pd.read_csv( "Absolute//C://Users//Administrator//Documents//GitHub//PYTHON//CSV_PROJECTS//Statistics_Probability_dataset_1//CSV" )

# Data analysis
num_rows, num_cols = dataset.shape
print( f"Number of Rows: { num_rows }" )
print( f"Number of Columns: { num_cols }" )

# Calculates descriptive statistics for each 
# column in the dataset
summary_stats = dataset.describe()
print( "Descriptive Statistics:" )
print( summary_stats )

# Probability calculation
if len( dataset ) >= 8:
    prob = norm.cdf( 205, loc = dataset[ 'column_1' ].mean(), scale = dataset[ 'column_1' ].std() )
    print( f"Probability: { prob }" )
else:
    print( "Insufficient Data for Probability Calculation." )  

# Perform Hypothesis Testing
# Ignore the warning for the small sample size
warnings.filterwarnings( "ignore" )
group_a_data = dataset[ dataset[ 'group' ] == 'A' ][ 'column_2' ]
group_b_data = dataset[ dataset[ 'group' ] == 'B' ][ 'column_2' ]

if len(group_a_data ) >= 8 and len( group_b_data ) >= 8:
    t_stat, p_value = ttest_ind( group_a_data, group_b_data )
    print( f"T_Statistic: { t_stat }" )
    print ( f"P-Value: { p_value }" )
else:
    print( "Insufficient Data for Hypothesis testing." )
    
# Correlation Analysis
numeric_columns = dataset.select_dtypes( include = [ np.number ]).columns
corr_matrix = dataset[ numeric_columns ].corr()
print( "Correlation Matrix:" )
print( corr_matrix )

# Perform Regression Analysis
if len( dataset ) >= 8:
    x = dataset[ [ 'clumn_1', 'column_2', 'feature_1', 'feature_2' ] ]
    y = dataset[ 'target' ]
    x = sm.add_constant( x )
    model = sm.OLS( y, x ).fit()
    print( model.summary() )
else:
    print( "Insufficient Data for Regression Analysis." )
    
     
    
    