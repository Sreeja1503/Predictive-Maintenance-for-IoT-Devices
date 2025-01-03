
# Predictive Maintenance using LSTM for Remaining Useful Life (RUL) Estimation

This project demonstrates how to use Long Short-Term Memory (LSTM) networks to predict the Remaining Useful Life (RUL) of machinery based on sensor data. The dataset used is from the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Predictive maintenance is a technique to predict when machinery will require maintenance before a failure occurs. This project uses LSTM networks to predict the RUL of machinery based on historical sensor data.

## Requirements
- Python 3.6 or higher
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

## Data Description
The dataset used in this project is from the CMAPSS dataset which contains multiple sensor readings for different units over time. The key columns in the dataset are:
- `unit_number`: Unique identifier for each unit.
- `time_in_cycles`: The time/cycle at which the data is recorded.
- `sensor_1` to `sensor_21`: Sensor readings.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/predictive-maintenance-lstm.git
   cd predictive-maintenance-lstm
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your dataset file (e.g., `train_FD001.txt`) in the `data` directory.

2. Run the script:
   ```bash
   python main.py
   ```

3. The script will:
   - Load and preprocess the data.
   - Normalize the sensor readings.
   - Calculate the Remaining Useful Life (RUL).
   - Create sequences from the time-series data.
   - Split the data into training and testing sets.
   - Build and train an LSTM model.
   - Evaluate the model and visualize the results.
   - Save the trained model to a file.

## Model Architecture
The LSTM model consists of:
- An LSTM layer with 100 units and ReLU activation.
- A dropout layer with a rate of 0.2.
- Another LSTM layer with 50 units and ReLU activation.
- Another dropout layer with a rate of 0.2.
- A dense layer with a single unit to predict the RUL.

## Results
The model's performance is evaluated using Mean Absolute Error (MAE) on the test set. The script also visualizes the true vs. predicted RUL.

## Conclusion
This project demonstrates a practical approach to predictive maintenance using LSTM networks for RUL estimation. By analyzing sensor data, the model can predict when machinery will require maintenance, helping to prevent unexpected failures and reduce downtime.

## References
- [CMAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

```
