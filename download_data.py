import os
import requests
import zipfile
import shutil
from argparse import ArgumentParser


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    parser = ArgumentParser()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    default_dataset_path = os.path.join(base_dir, 'processed_data')
    
    parser.add_argument("--store_path", type=str,
                        help="directory path to store processed data", default=default_dataset_path)
    args  = parser.parse_args()

    if os.path.exists(args.store_path):
        #remove the directory if exists
        print("Removing the existing directory...")
        shutil.rmtree(args.store_path)

    print("creating processed data directory")
    os.makedirs(args.store_path, exist_ok=True)

    print("Downloading Processed Dataset...")
    file_store_path = os.path.join(args.store_path, "processed_data_v4.zip")

    download_file_from_google_drive("11G8SC5ussSsGC-k7CeVrdsL-Xks8Aj4Q", file_store_path)

    with zipfile.ZipFile(file_store_path, 'r') as zfile:
        zfile.extractall(args.store_path)
    #finally delete the zip file
    os.remove(file_store_path)
    print("Downloaded successfully.")