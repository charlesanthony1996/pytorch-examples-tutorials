import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/employee_wellbeing_data.csv")

print(df.head())

# scaling the data
scaler = StandardScaler()
scaled_features = df[['Experience_years','Hours_Worked_per_week','Training_hours', 'Machine_Age_years', 'Maintenance_per_year','Worker_weight_kg','Worker_height_cm']]
scaled_features = scaler.fit_transform(scaled_features)

df_scaled = pd.DataFrame(scaled_features, columns=['Experiences_years', 'Hours_Worker_per_week','Training_hours', 'Machine_Age_years', 
'Maintenance_per_year','Worker_weight_kg','Worker_height_cm'])

df_scaled['Accident_yes_no'] = df['Accident_yes_no']

# checking for multilinear collinearity using VIF
vif_data = pd.DataFrame()
vif_data['feature'] = df_scaled.columns[:-1]
vif_data['VIF'] = [variance_inflation_factor(df_scaled.values, i) for i in range(df_scaled.shape[1] - 1)]
print(vif_data)




# fit a logistic regression model
# predictors = df[['Experience_years','Hours_Worked_per_week','Training_hours', 'Machine_Age_years', 'Maintenance_per_year','Worker_weight_kg','Worker_height_cm']]

# response = df_scaled['Accident_yes_no']

# predictors = sm.add_constant(predictors)
# logit_model = sm.Logit(response, predictors)
# result = logit_model.fit(maxiter=100)

# print(result.summary())


# fit a logistic regression model with l2 regularization
predictors = df[['Experience_years','Hours_Worked_per_week','Training_hours', 'Machine_Age_years', 'Maintenance_per_year','Worker_weight_kg','Worker_height_cm']]

response = df_scaled['Accident_yes_no']
logit_model = LogisticRegression(penalty='l2', max_iter=1000)
logit_model.fit(predictors, response)

# print the coefficients
print("intercept: ", logit_model.intercept_)
print("coefficient: ", logit_model.coef_)

# average accident rate
average_accident_rate = df['Accident_yes_no'].mean() * 100
print(f"The average accident rate: {average_accident_rate:.2f}%")


def predict_accident_risk(model, worker_data):
    # assuming worker is a data frame containing a single row of predictor values for a worker
    predicted_probability = model.predict_proba(worker_data)[:,1]
    return predicted_probability[0] * 100


worker_data = pd.DataFrame({
    'Experience_years': [5],
    'Hours_Worked_per_week': [40],
    'Training_hours': [10],
    'Machine_Age_years': [2],
    'Maintenance_per_year': [4],
    'Worker_weight_kg': [82],
    'Worker_height_cm': [170]
})

risk = predict_accident_risk(logit_model, worker_data)

print(f"Predicted accident risk for the worker: {risk:.2f}%")
