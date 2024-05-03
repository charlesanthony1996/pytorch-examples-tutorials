# # material usage over time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

# load the data
data = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")

# print(data.head())

data["Date"] = pd.to_datetime(data["Date"])

# plot material usage over time
# sns.lineplot(data = data, x="Date", y="Quantity", hue="Material_Type")
# plt.title("Material usage over time")
# plt.show()

# using plotly's api to add the features
fig = px.line(data, x="Date", y="Quantity", color="Material_Type", title="Material usage over time", 
            labels={"Quantity": "Usage_Quantity", "Date": "Date", "Material_Type":"Material_Type"}, 
            template="plotly_dark")

fig.update_xaxes(rangeslider_visible=True)

# add annotations (example: annotate maximum value, can be customized further)
max_point = data[data["Quantity"] == data["Quantity"].max()]
fig.add_annotation(x=max_point["Date"].values[0], y= max_point["Quantity"].values[0], text="Maximum", showarrow =True, arrowhead= 1)

fig.show()

#=============================

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
data = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")
data["Date"] = pd.to_datetime(data["Date"])  # Converting the 'Date' column to datetime

# Plotting material usage over time using Matplotlib and Seaborn
plt.figure(figsize=(10, 6))  # Set the figure size
sns.lineplot(data=data, x="Date", y="Quantity", hue="Material_Type", palette="tab10")  # Create a line plot

# Adding title and labels
plt.title("Material usage over time")
plt.xlabel("Date")
plt.ylabel("Usage Quantity")

# Annotate the maximum point
max_point = data[data["Quantity"] == data["Quantity"].max()]
plt.annotate('Maximum', xy=(max_point["Date"].values[0], max_point["Quantity"].values[0]), 
             xytext=(max_point["Date"].values[0], max_point["Quantity"].values[0] + 50), 
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()


# ===========================

# heatmap of material and product type

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# load the data 
df = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")

# print(df.tail())

heatmap_data = df.groupby(["Material_Type", "Product_Type"]).size().unstack()
sns.heatmap(heatmap_data, annot=True ,cmap="YlGnBu")
plt.title("Material vs product type")
plt.show()


# heatmap of material and product type using plotly
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")

# Heatmap data preparation
heatmap_data = df.groupby(["Material_Type", "Product_Type"]).size().unstack(fill_value=0)

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Viridis", cbar=True)

# Update layout
plt.title('Material vs Product Type')
plt.xlabel('Product Type')
plt.ylabel('Material Type')

# Show plot
plt.show()



# =========================


# histogram of bend angles (assuming bend angles are comma seperated)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")

# print(df.info())

# expanding the angles column
angles = data["Angles"].str.split(",", expand=True).stack()
sns.histplot(angles, bins=30)
plt.title("Histogram of bend angles")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample data loading
df = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")

# Expanding the angles column
angles = df["Angles"].str.split(",", expand=True).stack().astype(float)

# Create histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(angles, bins=30, color='#636EFA', edgecolor='black')

# Calculate median and add median line
median_angle = angles.median()
plt.axvline(median_angle, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_angle:.2f} degrees')

# Add labels, title, and legend
plt.xlabel('Bend Angles (degrees)')
plt.ylabel('Frequency')
plt.title('Histogram of Bend Angles')
plt.legend()

# Show plot
plt.grid(True)
plt.show()




# ==============================



# scatter plot for thickness vs quantity (proxy for wastage)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# print(df.head())

# sns.scatterplot(data = data, x="Thickness_mm", y="Quantity", hue="Material_Type")
# plt.title("Thickness vs Quantity")
# plt.show()


import pandas as pd
import plotly.express as px

# load the data (adjust the path as needed)
df = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/historical-material-data.csv")

# create scatter plot using plotly
fig = px.scatter(df,x="Thickness_mm", y="Quantity", color="Material_Type", title="Thickness vs Quantity",
 labels={"Thickness_mm": "Thickness (mm)", 'Quantity': 'Quantity ordered'}, hover_data=["Material_Type"])


# add reference lines if needed (average thickness and average quantity)
avg_thickness = df["Thickness_mm"].mean()
avg_quantity = df["Quantity"].mean()

fig.add_shape(type="line", x0=avg_thickness, x1=avg_thickness, y0=0, y1=df["Quantity"].max(),
line=dict(color="Red", width=2, dash="dash"))

fig.add_shape(type="line", x0=0, x1=df["Thickness_mm"].max(), y0=avg_quantity, y1=avg_quantity,
line=dict(color="Blue", width=2, dash="dash"))

fig.update_layout(hovermode="closest", template="plotly_dark")

fig.show()
