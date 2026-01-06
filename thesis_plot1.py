import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
parts_per_week = 1000

# Generate dummy data
data = []
np.random.seed(42)

for week in weeks:
    # Generate random probabilities for the width categories
    random_probs = np.random.rand(4)
    random_probs /= random_probs.sum()  # Normalize to sum to 1
    
    width_categories = ['Width ≤ 300mm', 'Width 300mm - 600mm', 'Width 600mm - 1000mm', 'Width > 1000mm']
    
    week_data = np.random.choice(
        width_categories,
        parts_per_week,
        p=random_probs
    )
    
    count_width_300mm = np.sum(week_data == "Width ≤ 300mm")
    count_width_300_600mm = np.sum(week_data == "Width 300mm - 600mm")
    count_width_600_1000mm = np.sum(week_data == "Width 600mm - 1000mm")
    count_width_above_1000mm = np.sum(week_data == "Width > 1000mm")

    data.append({
        "Week": week,
        "Width ≤ 300mm": count_width_300mm,
        "Width 300mm - 600mm": count_width_300_600mm,
        "Width 600mm - 1000mm": count_width_600_1000mm,
        "Width > 1000mm": count_width_above_1000mm,
        "Random Probabilities": random_probs
    })

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())

# Plotting the data
for i, row in df.iterrows():
    plt.figure(figsize=(10, 5))
    plt.bar(['Width ≤ 300mm', 'Width 300mm - 600mm', 'Width 600mm - 1000mm', 'Width > 1000mm'],
            [row['Width ≤ 300mm'], row['Width 300mm - 600mm'], row['Width 600mm - 1000mm'], row['Width > 1000mm']])
    plt.title(row['Week'])
    plt.ylabel('Number of Parts')
    plt.xlabel('Part Width')
    plt.show()
