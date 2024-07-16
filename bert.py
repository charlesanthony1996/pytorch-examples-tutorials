from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from helpers import get_gpu

# Get the GPU device
device = get_gpu()

# Define the file path (update this to the correct path if necessary)
data_dir = Path("/Users/charles/Desktop/data")
file_path = data_dir / "BBC News Sample Solution.csv"

# Check if the file exists and load the CSV file into a DataFrame
if file_path.is_file():
    data = pd.read_csv(file_path)
    print("File loaded successfully!")
else:
    print(f"File not found at {file_path}. Please check the file path.")
    data = pd.DataFrame()  # Empty DataFrame as a fallback

# Define the labels
labels = {"business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politics": 4}
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Print the directory for debugging
print(data_dir)

# Print the first few rows of the DataFrame (if loaded successfully)
if not data.empty:
    print(data.head())
else:
    print("Data frame is empty. Please check the file path and contents.")

