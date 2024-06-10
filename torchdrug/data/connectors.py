from azure.storage.blob import BlobServiceClient
import os


def get_blob_service_client(storage_account_name, 
                            storage_account_key):
    
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)

    return blob_service_client


STORAGE_ACCOUNT_NAME = os.environ['STORAGE_ACCOUNT_NAME']
STORAGE_ACCOUNT_KEY = os.environ['STORAGE_ACCOUNT_KEY']

def connect_to_blob_service_client():
    
    client = get_blob_service_client(STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY)
    return client