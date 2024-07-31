import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

glucose_hr = pd.read_csv("dataset")

# Split train data (90%) and test data (10%)

train_size = int(len(glucose_hr)*0.9)


glu_hr_train = glucose_hr[:train_size]
glu_hr_test = glucose_hr[train_size:]

def prepare_sequences(data, sequence_length):

    num_samples = len(data)
    X, y = [], []

    for i in range(num_samples - sequence_length + 1):
        if i+sequence_length<len(data):
            X.append(data[i:i+sequence_length, :])  # Extracting both heart rate and glycemia pairs
            y.append(data[i+sequence_length, 0])  # Extracting the next glycemia value

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y


seq_length = 12 

X_glu_hr_train, y_glu_hr_train = prepare_sequences(glu_hr_train, seq_length)
X_glu_hr_test, y_glu_hr_test = prepare_sequences(glu_hr_test, seq_length)

import tensorflow as tf
from tensorflow.keras.layers import Layer, GRU, Dense, Concatenate, Activation
from tensorflow.keras import initializers
from sklearn.metrics import mean_squared_error, mean_absolute_error

class AttentionLayer(Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        q = self.W(inputs)
        k = self.U(inputs)
        score = self.V(tf.nn.tanh(q + k))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class GRUWithAttention(tf.keras.Model):
    def __init__(self, units, attention_units):
        super(GRUWithAttention, self).__init__()
        self.gru = GRU(units, return_sequences=True)
        self.attention = AttentionLayer(attention_units)
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.gru(inputs)
        attention_output = self.attention(x)
        output = self.dense(attention_output)
        return output

units = 64
attention_units = 10
time_steps = 12
heart_rate_dim = 1  # Dimension of heart rate input
glycemia_dim = 1  # Dimension of glycemia output


#Build and compile the model
model = GRUWithAttention(units, attention_units)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_glu_hr_train, y_glu_hr_train, epochs=100, batch_size=32)
# Evaluate the model
loss = model.evaluate(X_glu_hr_train, y_glu_hr_train)
print("Test Loss:", loss)