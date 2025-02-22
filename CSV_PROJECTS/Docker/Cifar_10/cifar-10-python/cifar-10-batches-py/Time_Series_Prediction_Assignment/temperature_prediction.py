
#       ---- PROPHET ----
# NOTE: --> Prophet: Is a forecasting tool from the Prophet 
# library in Python used for time series analysis, 
# particularly for generating forecasts based on historical 
# data patterns. For instance, Prophet can be used by 
# businesses to predict future sales based on past sales data, 
# helping them optimize inventory and resource allocation.
# -->   scikit-learn: Is a machine learning library.
# -->   matplotlib: Is a plotting library in python that allows 
# creation of various visualizations.
#       ---- IMPORT_REQUIRED_LIBRARIES ----
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# NOTE: df is a common abbreviation used in programming, 
# particularly in Python with the pandas library. It (df) typically 
# stands for "DataFrame," which is a two-dimensional labeled data 
# structure with columns that can hold data of different types. 
df = pd.read_csv( 'daily_temperature.csv' )


# Convert the "Date (year, month, and day)", column to "datetime" 
# and sort the data. Converting a date to datetime is necessary when 
# time-related information or operations become relevant.
df[ 'Date' ] = pd.to_datetime( df[ 'Date' ] )
df.sort_values( 'Date', inplace=True )


# Preparing the dataset for Prophet
df_prophet = df.rename( columns={ 'Date': 'ds', 'Temperature': 'y' } )


# For the purpose of this assignment, select 
# the New York data only
df_prophet = df_prophet[ df_prophet[ 'City' ] == 'New York' ]


# Initialize and fit the model
model = Prophet( daily_seasonality=True)
model.fit( df_prophet )


# Make a Dataframe for predictions
# NOTE: Set the periods to "zero (0)", since I'm not 
# needing any future dates for this plot
future = model.make_future_dataframe( periods=0 )
forecast = model.predict( future )


# Merge the forecast with the original data
df_prophet.set_index( 'ds', inplace=True )
forecast.set_index( 'ds', inplace=True )
df_merged = df_prophet.join( forecast[ [ 'yhat', 'yhat_lower', 'yhat_upper' ] ], how='inner' )


# Reset the index for plotting purposes
df_merged.reset_index( inplace=True )


# Extract Actual and Predicted Temperatures
# Contains the actual temperature values
y_test = df_merged[ 'y' ]

# Contains the predicted temperature values
predictions = df_merged[ 'yhat' ]

mae = mean_absolute_error( y_test, predictions ) 
# NOTE: --> The function takes two arguments: y_test (actual values) 
# and predictions (predicted values).
# -->   MAE is the average of the absolute differences between actual 
# and predicted values.



# Plot the actual vs predicted temperatures
plt.figure( figsize=( 10, 6 ) )
plt.plot( df_merged[ 'ds' ], df_merged[ 'y' ], 'b-', label = 'Actual Temperature', marker = 'o', markersize = 8 )
plt.plot( df_merged[ 'ds' ], df_merged[ 'yhat' ], 'r-', label = 'Predicted Temperature', marker = 'o', markersize = 8 )


# -->   The result is formatted to 2 decimal 
# places for readability.
# Print the result
print( f"Mean Absolute Error (MAE): {mae:.2f}" )

# NOTE: Added code to calculate the MAE before the 
# for-loop. Why:
# a.    The MAE calculation is independent of the plotting 
# process. 
# b.    It's best to compute and print the MAE immediately 
# after merging the forecasted and actual data (df_merged). 
# c.    This ensures that any potential errors in the 
# calculations are caught before proceeding to the 
# actual and predicted visualizations.

for i, txt in enumerate( df_merged[ 'y' ] ):
    plt.annotate( round( txt, 2 ), ( df_merged[ 'ds' ][ i ], df_merged[ 'y' ][ i ] ), textcoords="offset points", xytext = ( 0, 10 ), ha = 'center' )

for i, txt in enumerate( df_merged[ 'yhat' ] ):
    plt.annotate( round( txt, 2 ), ( df_merged[ 'ds' ][ i ], df_merged[ 'yhat' ][ i ] ), textcoords="offset points", xytext = ( 0, 10 ), ha = 'center' )


# Code to run the functions
plt.legend()
plt.xlabel( 'Date' )
plt.ylabel( 'Temperature' )
plt.title( 'Actual vs Predicted Temperatures' )
plt.xticks( rotation = 45 )
plt.tight_layout()
plt.show()


