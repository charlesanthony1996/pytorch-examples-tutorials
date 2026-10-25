import os
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

base_dir = os.path.join("datasets", "aclImdb")