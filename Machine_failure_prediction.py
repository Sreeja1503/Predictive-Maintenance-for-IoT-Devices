import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Function to load data with error handling
def load_data(file_path, separator=" "):
    print("Loading data...")
    try:
        data = pd.read_csv(file_path, sep=separator, header=None)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")
        exit()
    except pd.errors.ParserError:
        print("Error: Could not parse the file. Please check the file format.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()

# Function to preprocess data
def preprocess_data(data, sensor_columns):
    print("Normalizing sensor data...")
    scaler = MinMaxScaler()
    data[sensor_columns] = scaler.fit_transform(data[sensor_columns])
    return data, scaler

# Function to calculate Remaining Useful Life (RUL)
def calculate_rul(df):
    print("Calculating Remaining Useful Life (RUL)...")
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df['RUL'] = max_cycle - df['time_in_cycles']
    print("RUL calculation completed.")
    return df

# Function to create sequences
def create_sequences(features, target, sequence_length=50):
    print(f"Creating sequences with sequence length: {sequence_length}")
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features.iloc[i:i + sequence_length].values)
        y.append(target.iloc[i + sequence_length])
    print("Sequence creation completed.")
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_lstm_model(input_shape):
    print("Building the LSTM model...")
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model

# Main script
if __name__ == "__main__":
    # Load data
    file_path = "CMAPSSData/train_FD001.txt"
    data = load_data(file_path)

    # Add column names
    print("Adding column names...")
    columns = ["unit_number", "time_in_cycles"] + [f"sensor_{i}" for i in range(1, 22)]
    data.columns = columns

    # Preprocess data
    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
    data, scaler = preprocess_data(data, sensor_columns)

    # Calculate RUL
    data = calculate_rul(data)

    # Feature selection
    print("Selecting features for modeling...")
    sensors_to_use = ["sensor_2", "sensor_3", "sensor_4"]  # Choose based on analysis
    features = data[sensors_to_use]
    target = data["RUL"]

    # Create sequences
    sequence_length = 50
    X, y = create_sequences(features, target, sequence_length)

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train LSTM model
    model = build_lstm_model((sequence_length, len(sensors_to_use)))
    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
    print("Model training completed.")

    # Evaluate the model
    print("Evaluating the model...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

    # Visualize predictions
    print("Generating predictions and visualizing results...")
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True RUL")
    plt.plot(y_pred, label="Predicted RUL")
    plt.legend()
    plt.show()

    # Save the model
    print("Saving the trained model...")
    model.save("lstm_predictive_maintenance.h5")
    print("Model saved successfully.")