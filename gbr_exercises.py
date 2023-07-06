import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# load the dataset
boston = datasets.fetch_california_housing()
print(boston)

california = datasets.fetch_california_housing()
x = california.data
y = california.target


# split the dataset into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# initialize the gradient boosting regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

print(gbr)

# fit the model to the training data
gbr.fit(x_train, y_train)

# predict the house prices on the test set
y_pred = gbr.predict(x_test)


# compute and print the root mean squared error of our predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("root mean squared error: ", rmse)