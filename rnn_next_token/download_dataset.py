import urllib.request
import os

# 1. Download the Tiny Shakespeare text dataset

file_name = "input.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, file_name)

if os.path.exists(file_name):
    print(f"File '{file_name}' already exists. Skipping download.")
    exit()
print("Downloading dataset...")
urllib.request.urlretrieve(url, file_name)
print("Download complete.")
