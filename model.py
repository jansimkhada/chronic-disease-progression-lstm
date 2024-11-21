import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1, activation='linear'))  # For regression problem (predicting progression)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(data):
    # Use only the relevant features for prediction (blood_pressure, cholesterol, age)
    X = data[['blood_pressure', 'cholesterol', 'age']].values
    y = data['disease_progression'].values

    # Reshape for LSTM input (samples, time_steps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = build_lstm_model(X_train.shape[1:])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save('lstm_model.h5')
    return model

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    data = preprocess_data('synthetic_medical_data.csv')
    model = train_model(data)
