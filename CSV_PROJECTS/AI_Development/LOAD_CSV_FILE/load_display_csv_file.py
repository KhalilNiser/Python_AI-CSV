
#       ---- IMPORT_REQUIRED_LIBRARY ----
# Pandas: Is a class in python library. Used for 
# data manipulation and analysis.
import pandas as pd


#       ---- LOAD_CSV_FILE ----
# This function reads the said CSV file and 
# loads it into a dataframe variable
data = pd.read_csv( "example_dataset_3.csv" )


#       ---- DISPLAY_FIRST_FIVE_ROWS ----
# The "data.head(5)", function retreives and returns 
# the first five rows of a dataset
print( data.head( 5 ) )