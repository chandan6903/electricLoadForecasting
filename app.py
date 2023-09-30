import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

demand_data = pd.DataFrame()

for folderYear in os.listdir("./loadData"):
    for folderMonth in os.listdir(f"./loadData/{folderYear}/"):
        for filename in os.listdir(f"./loadData/{folderYear}/{folderMonth}/"):
            if filename.endswith(".csv"):
                file_path = f"./loadData/{folderYear}/{folderMonth}/{filename}"
                temp_df = pd.read_csv(file_path)
                demand_data = demand_data._append(temp_df, ignore_index=True)

demand_data.sort_values(by="date", inplace=True)
demand_data.set_index("date", inplace=True)

print(demand_data)

demand_data = demand_data.reset_index()
print(demand_data.head())

plt.figure(figsize=(100, 20))
plt.plot(demand_data["Demand"])

data_training = pd.DataFrame(demand_data["Demand"][0: int(len(demand_data)*0.70)])
data_testing = pd.DataFrame(demand_data["Demand"][int(len(demand_data)*0.70): len(demand_data)])

print(data_training)
print(data_testing)

scaler = MinMaxScaler(feature_range = (0, 1))
data_training_array = scaler.fit_transform(data_training)
print(data_training_array)

x_train = []
y_train = []

for i in range(1440, data_training_array.shape[0]):
    x_train.append(data_training_array[i-1440: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()

model.add(LSTM(units = 50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10)

