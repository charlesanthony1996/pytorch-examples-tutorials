import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from prophet import Prophet

df = pd.read_csv("CO2_emission_wrangled.csv")

# shit works man!
# print(df.head(10))

# extract the data from austria
df_austria = df[df["Country_Name"] == "Austria"]
data = df_austria["CO2_emission"].values

mean_val = data.mean()

max_val = data.max()

min_val = data.min()

# stats summary
# print(df_austria)
# print(data)
# print(mean_val)
# print(max_val)
# print(min_val)

# define the lstm model
model = Sequential()
model.add(LSTM(units = 100, return_sequences=True, input_shape= (None, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units= 1))


print(model.summary())

model.compile(optimizer="adam", loss="mean_squared_error")

# prepare the data
x = data[:-1]
y = data[1:]

# x part of the dataset
# print(x)

# y part of the dataset
# print(y)

# normalize the data
min_val = np.min(data)
max_val = np.max(data)

x = (x - min_val) / (max_val - min_val)
y = (y - min_val) / (max_val - min_val)

print(x)

print(y)

# split the data into to the format and validation sets
train_size = int(len(x) * 0.8)
print(train_size)
x_train, x_val = x[:train_size], x[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
# so the rest is for validation -> 20%

# full count
full_count = int(len(x))
print(full_count)


# x train
print(x_train)

# y train
print(y_train)

# reshape the data to the format expected by lstm layers
x_train = x_train.reshape(-1, 1, 1)
x_val = x_val.reshape(-1, 1, 1)

# x train logging
print(x_train)

# x_val logging
print(x_val)

# fit the model to the data
history = model.fit(x_train, y_train, epochs= 100, batch_size=32, validation_data=(x_val, y_val))

# history logging
print(history)


# make a prediction for 2020 for austria 
last_value = data[-1]
last_value_normalized = (last_value - min_val) / (max_val - min_val)
prediction_normalized = model.predict(np.array([[last_value_normalized]]))
prediction = prediction_normalized * (max_val - min_val) + min_val
print("predicted co2 emission for austria in 2020 in lstm: ", prediction)


# make a prediction for 2020 for austria using the prophet model
df_austria_prophet = df_austria[["Year", "CO2_emission"]]
df_austria_prophet.columns = ["ds", "y"]
m = Prophet()
m.fit(df_austria_prophet)
future = m.make_future_dataframe(periods=1, freq ="Y")
forecast = m.predict(future)
print("Predicted co2 emission for austria in 2020 with prophet: ", forecast.tail(1)["yhat"])