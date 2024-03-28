# LSTM-Signal-Generation-

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to forecast a noisy sine wave signal.

# Dependencies

Python 3.x
NumPy
TensorFlow (2.x recommended)
Keras
matplotlib
pandas
scikit-learn (for StandardScaler)
Installation and Execution

# Install dependencies:

Bash
pip install numpy tensorflow keras matplotlib pandas scikit-learn

 Run the code:

Bash
python time_series_forecast.py 


 # Code Structure

Data Generation: Create a noisy sine wave signal as the time series data.
Data Preparation: Split the signal into overlapping sequences for input and target values.
Model Definition: Define the LSTM model architecture with an LSTM layer and a Dense output layer for making predictions.
Compilation: Choose an optimizer ('adam') and loss function ('mse') appropriate for this regression task.
Training: Train the LSTM model on the prepared data.
Visualization (Loss): Plot the training and validation loss over the epochs to monitor how well the model learns.
Forecasting: Iteratively use the trained model to generate forecasts one step ahead for the future time steps.
Visualization (Forecast): Plot the original noisy signal against the forecasted signal.

