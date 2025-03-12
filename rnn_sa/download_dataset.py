
import urllib.request
import tarfile
import os

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Define the output file name
file_name = "aclImdb_v1.tar.gz"
if os.path.exists(file_name):
    print(f"File '{file_name}' already exists. Skipping download.")
    exit()
print("Downloading dataset...")
urllib.request.urlretrieve(url, file_name)
print("Download complete.")

# Extract the dataset
print("Extracting dataset...")
with tarfile.open(file_name, "r:gz") as tar:
    tar.extractall()
print("Extraction complete.")

# Define dataset path
dataset_path = "aclImdb"

# Check the extracted structure
if os.path.exists(dataset_path):
    print(f"Dataset extracted successfully in '{dataset_path}'")
    print("Contents of the dataset directory:", os.listdir(dataset_path))
else:
    print("Dataset extraction failed.")
