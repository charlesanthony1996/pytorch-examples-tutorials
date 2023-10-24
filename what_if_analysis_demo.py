import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# sample data
data = {
    'Thickness_mm': [1, 2, 3, 4, 5],
    'Bend_Allowance_mm': [1.5, 2.5, 3.5, 4.5, 5.5],
    'Rejection_Rate': [0.1, 0.08, 0.07, 0.06, 0.05]
}

df = pd.DataFrame(data)

# print(df.head())

# split data into training and testing sets
x = df[['Thickness_mm', 'Bend_Allowance_mm']]
y = df['Rejection_Rate']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# create a linear regression 
# print(x_train)
# print(x_test)
model = LinearRegression()
model.fit(x_train, y_train)

# simulate what if scenarios
thickness_values = [i for i in range(1, 11)]
bend_allowance_values = [i* 0.5 for i in range(1, 21)]

predicted_rejected_rates = []

for t in thickness_values:
    for b in bend_allowance_values:
        predicted_rejected_rate = model.predict([[t, b]])[0]
        predicted_rejected_rates.append((t, b, predicted_rejected_rate))


# convert results into a dataframe for visualization
results_df = pd.DataFrame(predicted_rejected_rates, columns = ['Thickness_mm', 'Bend_Allowance_mm', 'Predicted_Rejection_Rate'])

print(results_df[:10])

# plotting the results -> just a basic visualization
plt.scatter(results_df['Thickness_mm'], results_df['Predicted_Rejection_Rate'], label='Thickness Impact', color='blue')
plt.scatter(results_df['Bend_Allowance_mm'], results_df['Predicted_Rejection_Rate'], label='Bend Allowance impact', color='red')
plt.xlabel("parameter value")
plt.ylabel("predicted rejection rate")
plt.title("what if analysis: impact on rejection rate")

# plt.show()