import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


crops = pd.read_csv("/users/charles/downloads/soil_measures.csv")
print(crops.head())

y = crops["crop"]
x = crops.drop("crop", axis=1)

# placeholder for the best feature and its score
best_score = 0
best_feature = None

# iterate over each feature
for feature in x.columns:
    x_feature = x[[feature]]

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y, test_size = 0.3, random_state=42)


    # initialize and train the logistic regression model
    # model = LinearRegression()
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # predict and evaluate the model
    y_pred = model.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)

    if score > best_score:
        best_score = score
        best_feature = feature


# create the dictionary
best_predictive_feature = { best_feature: best_score}
print(best_predictive_feature)