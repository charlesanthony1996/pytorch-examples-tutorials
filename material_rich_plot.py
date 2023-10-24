import torch
import tensorflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# doing a cost analysis on the historical data
df = pd.read_csv("historical-material-rich-data.csv")

# print(df.head())

# can relief cuts be a boolean value? are the numbers useful to us? please search

# assumptions

# cost is in euros
# only two materials for now
# rejected rates are imagined


def safe_float_conversion(x):
    try:
        # return float(x)
        # return float(x('%')) / 100.0
        return float(x.strip('%')) / 100.0
    except ValueError:
        return np.nan


def convert_to_float(value):
    try:
        return float(value.strip('mm'))
    except:
        return np.nan

columns_to_convert = ['Bend_Radius_Tolerance', 'Flat_Dimensions_Tolerance']
for column in columns_to_convert:
    # print(df[column])
    df[column] = df[column].apply(convert_to_float)


df['Rejection_Rate'] = df['Rejection_Rate'].apply(safe_float_conversion)

df['Rejection_Rate'].fillna(0, inplace=True)

# calculate the cost due to rejected items
df['Rejected_Cost'] = df['Rejection_Rate'] * df['Total_Cost']

print(f"Total rejected cost: ",df['Rejected_Cost'].sum() , "%" )

# aggregate the total costs 
agg_data = df.groupby('Material_Type').agg(Total_Cost = pd.NamedAgg(column='Total_Cost', aggfunc='sum'),
                                            Rejected_Cost = pd.NamedAgg(column = 'Rejected_Cost', aggfunc='sum')).reset_index()


# print(agg_data)

# calculate the percentage of cost wasted due to rejections for each material type
agg_data['Wastage_Percentage'] = (agg_data['Rejected_Cost'] / agg_data['Total_Cost'] * 100)

# print(agg_data)

# identify materials with wastage cost higher than a certain threshold
high_wastage_materials = agg_data[agg_data['Wastage_Percentage'] > 5]


# print(high_wastage_materials)

# advanced visualization

# cost distribution by material type
plt.figure(figsize=(12, 6))
sns.barplot(x='Material_Type', y='Total_Cost', data=agg_data, palette='Blues_d')
plt.title("total cost by material type")
plt.ylabel("total cost")
plt.xlabel("material type")
# plt.show()


# wasted cost due to rejection by material type
plt.figure(figsize=(12, 6))
sns.barplot(x='Material_Type', y='Rejected_Cost', data=agg_data, palette='Reds_d')
plt.title("wasted cost due to rejections by material type")
plt.ylabel("rejected cost") 
plt.xlabel("material type")
# plt.show()



# heatmap to visualize correlation between factors
correlation_matrix = df[['Thickness_mm','Total_Cost', 'Rejection_Rate', 'Bend_Allowance_mm', 'Bend_Radius_Tolerance', 'Flat_Dimensions_Tolerance']].corr()

print(correlation_matrix)
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix,annot=True, cmap='coolwarm')
plt.title("correlation heatmap")
# plt.show()

print(f"materials with high wastage cost (5%)",high_wastage_materials['Material_Type'].tolist())

# explanation on the matrix

# Thickness_mm vs Bend_Allowance_mm -> there is a very high positive correlation between these two indicating that
# as the thickness of the material increases , the bend allowance also tends to increase.

# Thickness_mm vs Bend_Radius_Allowance and Thickness_mm vs Flat_Dimensions_Tolerance -> both of these variables have a
# position correlation of 0.95 with Thickness_mm, indicating a strong positive linear relationship . As the thickness increases,
# both the bend radius tolerance and the flat dimensions tolerance also tend to increase

# Total_Cost vs Rejection_Rate -> a correlation of 0.34 , indicating a weak positive relationship.
# this suggests that as the total cost increases , the rejection rate might also slightly increase, though the relationship
# is not very strong

#  Rejection_Rate vs Bend_Allowance_mm -> A correlation of 0.73 indicates a moderately positive relationship
# higher rejection rates are associated with greater bend allowances

# Bend_Radius_Tolerance vs Flat_Dimensions_Tolerance -> the correlation 0.99, signifying a very strong positive
# linear relationship . so they tend to vary, and adjustemnts to one another might impact each other
