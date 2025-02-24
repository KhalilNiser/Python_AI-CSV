
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# NOTE: Pands: Is a powerful library in python 
# used for data manipulation and analysis
# Numpy: Is a library in python used for 
# numerical computing
import pandas as pd
import numpy as np


# Load the dataset: Reads the dataset from a CSV 
# file and loads it into a Pandas DataFrame I 
# created/named "df"
file_path = "example_dataset.csv"
df = pd.read_csv( file_path )

#       ---- BASIC_DATA_CLEANING ----
# Handle missing values by filling them with 
# the MEAN of the corresponding column. "df.mean()": 
# Calculates the MEAN for each column. 
# "df.fillna(..., inplace=True)": Ensures that the 
# changes are being applied directly to the DataFrame
df.fillna( df.mean(), inplace=True )

# Remove duplicate rows
# "df.drop_duplicates()": Removes duplicate rows from 
# the dataset. 
# "inplace=True" Modifies the original DataFrame instead 
# of creating a new one
df.drop_duplicates( inplace=True )

#       ---- FEATURE_ENGINEERING_(CREATING_NEW_FEATURES) ----
# Create new interaction features. 
# Example: Sum of all features. Adds a new 
# column (sum_features). Which contains the sum of 
# all the features in each row. 
# "df.sum(axis=1)": Computes the row-wise sum
df[ 'feature_sum' ] = df.sum( axis=1 )

# Example: Product of the first two features
# Creates a new column (product_feature_1_2). 
# Which is the multiplication of both feature_1 
# and feature_2
df[ 'feature_product' ] = df[ 'feature_1' ] * df[ 'feature_2' ]

# Example: Mean of all the features
# Adds a new column (feature_mean). Which the 
# mean of all feature values of each row
df[ 'feature_mean' ] = df.iloc[ :, :-2 ].mean( axis=1 ) 

# Save the cleaned and modified dataset 
# (cleaned_dataset.csv). 
# "index=False": Prevents Pandas from writing 
# indices into the file. 
# NOTE: By default whenever we save a DataFrame 
# using "VariableName.to_csv()", Pandas adds an 
# extra index column (which is the row numbers). 
# To not have that extra column I write "index=False"
df.to_csv( "preprocessed_dataset.csv", index=False)

# Display the first few rows
print( df.head() )