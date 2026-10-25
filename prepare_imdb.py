import os
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

base_dir = os.path.join("datasets", "aclImdb")

for split_name in ["train", "test"]:

    split_data = dataset[split_name]

    for index, example in enumerate(split_data):
        text = example["text"]

        label = example["label"]

        if label == 1:
            label_folder = "pos"

        else:
            label_folder = "neg"


        folder_path = os.path.join(base_dir, split_name, label_folder)
        os.makedirs(folder_path, exist_ok=True)

        file_name = str(index) + ".txt"

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    print("Finished writing", split_name)

