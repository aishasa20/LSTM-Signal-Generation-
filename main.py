import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import  StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

x = np.linspace(0, 10 * np.pi, 1000)

signal = np.sin(x) + 0.3 * np.random.randn(1000)

plt.plot(signal)
plt.show()

sequence_length = int(2* np.pi)
X = []
y = []
for i in range(len(signal) - sequence_length):
    X.append(signal[i:i+sequence_length])
    y.append(signal[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(None, 1)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()

history = model.fit(X, y, epochs = 30, batch_size = 16, validation_split = 0.2, verbose = 1)

# Loss curve
plt.plot(history.history['loss'], label= "Training Loss")
plt.plot(history.history['val_loss'], label= "Validation Loss")
plt.legend()
plt.show()

# Forecast series for next sequence length 

forecast = signal[:sequence_length].tolist()

for i in range(len(signal) - sequence_length):
    input_sequence = np.array(forecast[-sequence_length:]).reshape(1, sequence_length, 1)
    predicted_value = model.predict(input_sequence)
    forecast.append(predicted_value[0, 0])

# Forecasted vs. Actual Signal

plt.figure(figsize=(10, 6))
plt.plot(signal, label='Noisy Signal')
plt.plot(x[:len(forecast)], forecast, label='Forecast')
plt.show()
