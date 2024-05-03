import pandas as pd

# Load data directly into a DataFrame
data_url = "https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/labeled_data.csv"
data = pd.read_csv(data_url)


data['tweet'] = data['tweet'].str.lower()


data.to_csv('output_data_hs.csv', index=False)

print("CSV file has been created successfully.")
