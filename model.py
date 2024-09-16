# Data preparation
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, Activation, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error

glucose_hr = pd.read_csv("dataset")
glucose_hr = glucose_hr.iloc[:, [1, 2]].values #to take only Blood Glucose and Heart Rate values

# Split train data (90%) and test data (10%)
train_size = int(len(glucose_hr)*0.9)
glu_hr_train = glucose_hr[:train_size]
glu_hr_test = glucose_hr[train_size:]

glu_hr_train = glu_hr_train.astype(int)
glu_hr_train = glu_hr_train.astype(float)
glu_hr_test = glu_hr_test.astype(int)
glu_hr_test  = glu_hr_test.astype(float)

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

seq_length = 12 #choose the length of the time window of measured values to be considered
X_glu_hr_train, y_glu_hr_train = prepare_sequences(glu_hr_train, seq_length)
X_glu_hr_test, y_glu_hr_test = prepare_sequences(glu_hr_test, seq_length)


X_glu_train = X_glu_hr_train[:,:,0] #to take only the blood glucose values
X_hr_train = X_glu_hr_train[:,:,1] #to take only the heart rate values

#Model

input_shape_glu = (12, 1)  # # Input shape Blood glucose values
input_shape_hr = (12, 1)   # # Input shape Heart rate values

hidden_units = 64 # Number of hidden units for the GRU layers

# Input layers
inputs_glu = Input(shape=input_shape_glu, name='glucose_input')
inputs_hr = Input(shape=input_shape_hr, name='heart_rate_input')

# GRU layers
gru_output_glu = GRU(units=hidden_units, return_sequences=True)(inputs_glu)
gru_output_hr = GRU(units=hidden_units, return_sequences=True)(inputs_hr)

concatenated_output = Concatenate(axis=-1)([gru_output_glu, gru_output_hr]) # Concatenate GRU outputs

# Temporal attention mechanism
temporal_attention = Dense(1, activation='tanh')(concatenated_output)
temporal_attention = Activation('softmax')(temporal_attention)
att_weighted_glu = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=-1))([temporal_attention, gru_output_glu])
att_weighted_hr = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=-1))([temporal_attention, gru_output_hr])

concatenated_att_outputs = Concatenate(axis=-1)([att_weighted_glu, att_weighted_hr]) # Concatenate attention-weighted outputs

output = Dense(1)(concatenated_att_outputs) # Output layer

model = Model(inputs=[inputs_glu, inputs_hr], outputs=output) # Define the model

model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model

model.fit({'glucose_input': X_glu_train, 'heart_rate_input': X_hr_train}, y_glu_hr_train, epochs=20, batch_size=32) # Train the model

# Compute Mean Squared Error (MSE) and Mean Absolute Error (MAE)
X_glu_test = X_glu_hr_test[:,:,0]
X_hr_test = X_glu_hr_test[:,:,1]
predictions = model.predict({'glucose_input': X_glu_test, 'heart_rate_input': X_hr_test})
mse = mean_squared_error(y_glu_hr_test, predictions)
print("Mean Squared Error: ", mse)

mae = mean_absolute_error(y_glu_hr_test, predictions)
print("Mean Absolute Error: ", mae)
