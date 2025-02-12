
#               ---- IMPORT_REQUIRED_LIBRARIES ----
import os
from google.cloud import storage
from google.cloud import aiplatform

# Setup Project and Region Variables
PROJECT_ID = "inventorymanager-450120"
REGION = "us-central1"

# Set the Bucket_Name
BUCKET_NAME = "khalil-new-bucket-987654321"

# Authenticate Google Cloud Account
os.environ[ "GOOGLE_APPLICATION_CREDENTIALS" ] = "inventorymanager-450120-ec834d4c0c0c.json"

# Function create_bucket(bucket_name): Takes-in a parameter as an argument 
# "bucket_name". Creates a Cloud Storage Bucket
def create_bucket( bucket_name ):
    """
    Creates a Cloud Storage Bucket with the given name
    """
    
    storage_client = storage.Client()
    bucket = storage_client.create_bucket( bucket_name )
    print( "Bucket Created:", bucket.name )

# Uploads a Dataset file to the specified Cloud Storage Bucket
def upload_dataset( bucket_name, dataset_path ):
    """
    Uploads a Dataset file to the specified Cloud Storage Bucket
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket( bucket_name )
    blob = bucket.blob( os.path.basename( dataset_path ) )
    blob.upload_from_filename( dataset_path )
    print( "Dataset File Uploaded to Cloud Storage. Successfully!" )
    
# Create the Tabular Dataset to Vertex AI
def create_tabular_dataset( bucket_name, dataset_path ):
    
    """
    Creates a Tabular Dataset in Vertex AI using the 
    specified bucket and dataset file
    """
    aiplatform.init( project=PROJECT_ID, location=REGION )
    
    gcs_source_uri = f"gs://{ bucket_name } / { os.path.basename( dataset_path ) }"
    dataset = aiplatform.TabularDataset.create( display_name="my_dataset", gcs_source=gcs_source_uri )
    print( "Tabular Dataset Created:", dataset.display_name )
    
# Analyze the Dataset
def analyze_dataset( bucket_name, dataset_path ):
    dataset = aiplatform.TabularDataset(
        project = PROJECT_ID,
        location = REGION,
        display_name = "my_dataset",
        gcs_source = f"gs://{ bucket_name } / { os.path.basename( dataset_path ) }"
    )
    analysis = dataset.analyze()
    print( "Dataset Analysis Results:" )
    print( analysis )

    
# Main Function
def main():
    """
    Main function that orchestrates the steps to create 
    the bucket, upload the dataset, and create the 
    Tabular Dataset
    """
    
    create_bucket( BUCKET_NAME )
    upload_dataset( BUCKET_NAME, r"C:\\Users\\Administrator\\Documents\\GitHub\\PYTHON\\CSV_PROJECTS\\CSV_Assignment\\images\\dogs\\dog.jpg,dogs" )
    create_tabular_dataset( BUCKET_NAME, r"C:\\Users\\Administrator\\Documents\\GitHub\\PYTHON\\CSV_PROJECTS\\CSV_Assignment\\images\\dogs\\dog.jpg,dogs" )
    
# Execute the main function
if __name__ == "__main__":
    main()


