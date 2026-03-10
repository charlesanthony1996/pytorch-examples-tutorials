import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load dataset
data = load_iris()
x = data.data
y = data.target

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

# base line centralized

baseline_model = LogisticRegression(max_iter = 200)
baseline_model.fit(x_train, y_train)

baseline_pred = baseline_model.predict(x_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("centralized accuracy: ", baseline_acc)

# federated learning simulation
num_clients = 3
client_data = np.array_split(x_train, num_clients)
client_labels = np.array_split(y_train, num_clients)

client_models = []

# each client trains locally
for i in range(num_clients):
    model = LogisticRegression(max_iter = 200)
    model.fit(client_data[1], client_labels[i])
    