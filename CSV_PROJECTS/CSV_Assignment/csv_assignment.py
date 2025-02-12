
# Import Necessary Libraries
import os
import pandas as pd

# Root Directory of Image dataset
dataset_root = "C:\\Users\\Administrator\\Documents\\GitHub\\PYTHON\\CSV_PROJECTS\\CSV_Assignment\\images\\dogs"

# Initialize an empty DataFrame
image_data = pd.DataFrame( columns = [ "image_path", "label" ] )

# Traverse the dataset directory
for root, dirs, files in os.walk( dataset_root ):
    # Iterate over each file in the current directory
    for file in files:
        # Combine the root directory and file name 
        # to get the complete image path
        image_path = os.path.join( root, file )
        
        # Extract the label or class name from the directory name
        label = os.path.basename( root )
        
        row = pd.DataFrame( { "image_path": [ image_path ], "label": [ label ] } )
        
        # Concatenate the new row with the existing image_data DataFrame
        image_data = pd.concat( [ image_data, row ], ignore_index=True )
        
image_data.to_csv("image_dataset.csv", index=False)

file_path = "image_dataset.csv"

os.startfile( file_path )