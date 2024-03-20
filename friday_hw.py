import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression

diabetes_ds = datasets.load_diabetes()

print(diabetes_ds)

diabetes_df = pd.DataFrame(data=diabetes_ds.data, columns=diabetes_ds.feature_names)


# continuous target variable -> KNeighborsRegressor

# feature engineering can be performed here -> but not done here

diabetes_df["target"] = diabetes_ds.target
print(diabetes_df)

# dataset sample

# you can comment this out for the last plot to see the total instances
diabetes_df = diabetes_df.sample(n=150, random_state=42)

# create a dictionary with the number 
label_distribution = {str(label): list(diabetes_df.target).count(label) for label in set(diabetes_df.target)}
labels = list(label_distribution.keys())
values = list(label_distribution.values())
print("Class distribution (class: no of samples): ", label_distribution)

fig = plt.figure()

# creating the bar plot
plt.bar(labels, values)

plt.xlabel("label")
plt.ylabel("No of samples")
plt.title("Dataset distribution")
plt.show()

print(diabetes_df.corr())


from sklearn.model_selection import train_test_split


# why did we pick these columns?
x = diabetes_df[['age', 'sex', 'bmi', 's4']]
y = diabetes_df['target']

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# model = KNeighborsClassifier()
model = KNeighborsRegressor()

model_multiclass = model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_pred)

# Mean Absolute Error (MAE): The average of the absolute differences between the predictions and actual values. 
# It gives an idea of how wrong the predictions were.

# Mean Squared Error (MSE): The average of the squared differences between the predictions and actual values. 
# It penalizes larger errors more than MAE.

# R-squared (R²): Represents the proportion of the variance for the dependent variable 
# that's explained by the independent variables in the model. Essentially, it provides 
# a measure of how well observed outcomes are replicated by the model.


from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# print(metrics.classification_report(y_test, y_pred))

# conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# sns.heatmap(pd.DataFrame(conf_matrix), cmap='YlGnBu', annot=True, fmt='g')

# Instead of using a confusion matrix (which is for classification tasks), 
# it is better to visualize the regression model's performance by plotting 
# the actual vs. predicted values to see how closely they align.

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual value vs Predicted value")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# The closer the points are to the diagonal line, the better the predictions.


# results on what the metrics mean

# Mean Absolute Error (MAE) of 47.08 suggests that, on average, 
# the predictions of the KNeighborsRegressor model are about 47.08 units away 
# from the actual values. Given the context of the diabetes dataset, 
# this number should be evaluated relative to the range of the target variable to determine its significance.


# Mean Squared Error (MSE) of 3241.25 indicates the average of the squares of the errors. 
# This value is quite sensitive to outliers and should be considered in the context of the dataset's target range.

# R-squared (R²) of 0.3839 reflects that approximately 38.39% of the variance in the 
# dependent variable is predictable from the independent variables. While not particularly high, 
# it offers a starting point for model improvement.