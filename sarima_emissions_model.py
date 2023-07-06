from tinygrad.tensor import Tensor
import numpy as np
from keras.models import Sequential
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from keras.layers import LSTM, Dense, Dropout


df = pd.read_csv("CO2_emission_wrangled.csv")

# extract the data for austria
df_austria = df[df["Country_Name"] == "Austria"]
data = df_austria["CO2_emission"].values

print(data)


# define a more complex lstm model
model = Sequential()

model.add(LSTM(units = 100, return_sequences=True, input_shape=(None, 1)))
# why is dropoiut 0.2?
# learn what other models there are though?
# redo this with a cnn or rcnn
model.add(Dropout(0.2))
model.add((LSTM(units= 50, return_sequences=True)))
model.add(Dropout(0.2))
model.add((LSTM(units = 50)))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

x = model.compile(optimizer="adam", loss="mean_squared_error")
print(x)


# prepare the data
x = data[:-1]
y = data[1:]

print(x)
print(y)

# normalize the data
min_val = np.min(data)
max_val = np.max(data)

x = (x - min_val)  / (max_val - min_val)
y = (y - max_val) / (max_val - min_val)

# split the data into training and valiation sets
train_size = int(len(x) * 0.8)
print(train_size)

full_item_length = int(len(x))
print(full_item_length)

x_train, x_val = x[:train_size], x[train_size:]
y_train, y_val = y[:train_size], y[train_size:]


# reshape the data to the format expected by lstm layers
x_train = x_train.reshape(-1, 1, 1)
x_val = x_val.reshape(-1, 1, 1)


# fit the model to the data
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

print(history)

# make a prediction for austria in 2020
last_value = data[-1]
last_value_normalized = (last_value - min_val) / (max_val - min_val)
prediction_normalized = model.predict(np.array([[last_value_normalized]]))
prediction = prediction_normalized * (max_val - min_val) + min_val
print("prediction: ", prediction)
print("predicted co2 emission for austria in 2020 with lstm: ", prediction)


# now try it with sarima
model_sarima = SARIMAX(data, order=(1, 1, 1), seasonal1_order= (1, 1, 1, 1))
model_sarima_fit = model_sarima.fit(disp=False)
forecast_sarima = model_sarima_fit.predict(len(data), len(data))
print("forecast sarima: ", forecast_sarima)

print("predicted c02 emission for austria in 2020 with sarima: ", forecast_sarima)


