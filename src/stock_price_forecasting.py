# import filtered stock price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load data
data = pd.read_csv('data/stock_prices/AAPL_data_20250409.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Split data into train and test sets (80% train, 20% test)
train_ratio = 0.8
train_size = int(len(data) * train_ratio)
train_data = data[:train_size]
test_data = data[train_size:]

print(f"Training data: {train_data.shape[0]} rows")
print(f"Test data: {test_data.shape[0]} rows")

# Visualize the stock open price data with the train/test split
plt.figure(figsize=(14, 6))
plt.plot(data['Date'], data.iloc[:, 2], color='blue', label='Stock Open Price')
plt.axvline(x=train_data['Date'].iloc[-1], color='r', linestyle='--', 
            label=f'Train/Test Split ({train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%)')  # mark the training-test split timepoint
# Convert Date to datetime and set ticks at 3-month intervals
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.title('Historical Stock Open Price with Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Extract and scale the features
# For training set
train_set = train_data.iloc[:, 2:3].values
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)

# For test set (using the same scaler)
test_set = test_data.iloc[:, 2:3].values
test_set_scaled = sc.transform(test_set)

# Create training data structure for hyperparameter tuning
X_train = []
y_train = []

# Enhanced hyperparameter tuning for timesteps (lagging days)
timesteps_range = [15, 30, 45, 60, 75, 90, 120]
best_mse = float('inf')
best_timesteps = 60  # initial guess
# Store results for each timestep
timesteps_results = {}

# Further split training data for validation during hyperparameter tuning
val_ratio = 0.2
val_size = int(len(train_set_scaled) * (1 - val_ratio))
train_val_data = train_set_scaled[:val_size]
validation_data = train_set_scaled[val_size:]

for timesteps in timesteps_range:
    X_temp = []
    y_temp = []
    for i in range(timesteps, len(train_val_data)):
        X_temp.append(train_val_data[i-timesteps:i, 0].tolist())
        y_temp.append(train_val_data[i, 0])
    
    X_validation = []
    y_validation = []
    for i in range(timesteps, len(validation_data)):
        X_validation.append(validation_data[i-timesteps:i, 0])
        y_validation.append(validation_data[i, 0])
    
    X_temp = np.array(X_temp)
    y_temp = np.array(y_temp)
    X_temp = np.reshape(X_temp, (X_temp.shape[0], X_temp.shape[1], 1))
    
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
    
    # Use a simple model for testing
    temp_model = Sequential([
        LSTM(units=50, input_shape=(timesteps, 1)),
        Dense(units=1)
    ])
    temp_model.compile(optimizer='adam', loss='mean_squared_error')
    temp_model.fit(X_temp, y_temp, epochs=5, batch_size=32, verbose=0)
    
    # Evaluate on validation set
    val_mse = temp_model.evaluate(X_validation, y_validation, verbose=0)
    train_mse = temp_model.evaluate(X_temp, y_temp, verbose=0)
    
    timesteps_results[timesteps] = {'train_mse': train_mse, 'val_mse': val_mse}
    
    if val_mse < best_mse:
        best_mse = val_mse
        best_timesteps = timesteps

# Visualize the timestep selection results
plt.figure(figsize=(10, 6))
train_mse_values = [timesteps_results[t]['train_mse'] for t in timesteps_range]
val_mse_values = [timesteps_results[t]['val_mse'] for t in timesteps_range]
plt.plot(timesteps_range, train_mse_values, 'o-', label='Training MSE')
plt.plot(timesteps_range, val_mse_values, 'o-', label='Validation MSE')
plt.axvline(x=best_timesteps, color='r', linestyle='--', label=f'Best Timesteps = {best_timesteps}')
plt.xlabel('Timesteps (Lagging Days)')
plt.ylabel('Mean Squared Error')
plt.title('Timestep Selection Results')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best timesteps found: {best_timesteps} with validation MSE: {best_mse:.6f}")

# Use the best timesteps for final training data on the full training set
X_train = []
y_train = []
for i in range(best_timesteps, len(train_set_scaled)):
    X_train.append(train_set_scaled[i-best_timesteps:i, 0])
    y_train.append(train_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# build RNN model with dynamic input shape based on best_timesteps
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(best_timesteps, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# evaluate model performance
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Training Loss: {train_loss}')

# make predictions on training data
train_predictions = model.predict(X_train)
train_predictions = sc.inverse_transform(train_predictions)
actual_prices = sc.inverse_transform([y_train])

# Apply trend adjustment to fix edge problem
# Calculate average error to see if predictions are systematically off
error = actual_prices[0] - train_predictions[:,0]
mean_error = np.mean(error)
print(f"Mean prediction error: {mean_error:.4f}")

# Apply bias correction to training predictions
train_predictions_adjusted = train_predictions + mean_error

# Visualize the effect of adjustment
plt.figure(figsize=(12, 6))
plt.plot(train_data.iloc[best_timesteps:]['Date'], actual_prices[0], 'r-', label='Actual Prices')
plt.plot(train_data.iloc[best_timesteps:]['Date'], train_predictions[:,0], 'b--', label='Original Predictions')
plt.plot(train_data.iloc[best_timesteps:]['Date'], train_predictions_adjusted[:,0], 'g-', label='Adjusted Predictions')
plt.title('Effect of Trend Adjustment on Training Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# calculate metrics for original and adjusted predictions
mse = mean_squared_error(actual_prices[0], train_predictions[:,0])
r2 = r2_score(actual_prices[0], train_predictions[:,0])
adjusted_mse = mean_squared_error(actual_prices[0], train_predictions_adjusted[:,0])
adjusted_r2 = r2_score(actual_prices[0], train_predictions_adjusted[:,0])

print(f'Original MSE: {mse:.2f}, R2 Score: {r2:.2f}')
print(f'Adjusted MSE: {adjusted_mse:.2f}, R2 Score: {adjusted_r2:.2f}')

# Prepare test data sequence - FIXING THE DIMENSION ERROR
X_test = []
y_test = []

# Create a single continuous sequence for prediction by combining train and test data
combined_data = np.vstack((train_set_scaled, test_set_scaled))

# Generate sequences that span the boundary between train and test sets
X_test = [combined_data[i:i+best_timesteps, 0] for i in range(len(train_set_scaled) - best_timesteps, len(combined_data) - best_timesteps)]
y_test = [combined_data[i+best_timesteps, 0] for i in range(len(train_set_scaled) - best_timesteps, len(combined_data) - best_timesteps)]

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# make predictions on test data
test_predictions = model.predict(X_test)
test_predictions = sc.inverse_transform(test_predictions)

# Apply the same adjustment to test predictions
test_predictions_adjusted = test_predictions + mean_error

# The actual test prices correspond to our predictions, but shifted by the timesteps
# We need to align the predictions with the original test data
actual_test_prices = test_set[:len(test_predictions)]

# calculate test metrics (only if shapes match)
if len(actual_test_prices) == len(test_predictions):
    test_mse = mean_squared_error(actual_test_prices, test_predictions)
    test_r2 = r2_score(actual_test_prices, test_predictions)
    adjusted_test_mse = mean_squared_error(actual_test_prices, test_predictions_adjusted)
    adjusted_test_r2 = r2_score(actual_test_prices, test_predictions_adjusted)
    
    print(f'Test MSE: {test_mse:.2f}, R2 Score: {test_r2:.2f}')
    print(f'Adjusted Test MSE: {adjusted_test_mse:.2f}, R2 Score: {adjusted_test_r2:.2f}')
else:
    print(f'Shape mismatch: actual_test_prices {actual_test_prices.shape}, test_predictions {test_predictions.shape}')

# plotting final predictions with adjustments
plt.figure(figsize=(15,6))

# Plot training predictions
plt.subplot(1, 2, 1)
plt.plot(train_data.iloc[best_timesteps:]['Date'], actual_prices[0], color='red', label='Real Stock Price')
plt.plot(train_data.iloc[best_timesteps:]['Date'], train_predictions_adjusted[:,0], color='blue', label='Adjusted Predicted Price')
plt.title('Stock Price Prediction - Training Set (Adjusted)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)

# Plot test predictions
plt.subplot(1, 2, 2)
test_dates = test_data['Date'][:len(test_predictions)]
plt.plot(test_data['Date'][:len(actual_test_prices)], actual_test_prices, color='red', label='Real Stock Price')
plt.plot(test_dates, test_predictions_adjusted, color='blue', label='Adjusted Predicted Price')

# Add the boundary marker to this plot as well
if best_timesteps < len(test_dates):
    forecast_boundary_date = test_dates.iloc[0] + pd.Timedelta(days=best_timesteps)
    plt.axvline(x=forecast_boundary_date, color='purple', linestyle='-.', linewidth=2,
                label=f'Pure Forecast Start (t+{best_timesteps})')

plt.title('Stock Price Prediction - Test Set (Adjusted)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Compare original vs adjusted predictions in the test set
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_test_prices, 'r-', label='Actual Prices')
plt.plot(test_dates, test_predictions, 'b--', label='Original Predictions')
plt.plot(test_dates, test_predictions_adjusted, 'g-', label='Adjusted Predictions')

# Mark the point after which no training data is used in forecasting
if best_timesteps < len(test_dates):
    forecast_boundary_date = test_dates[best_timesteps]
    plt.axvline(x=forecast_boundary_date, color='purple', linestyle='-.', linewidth=2,
                label=f'Pure Forecast Boundary (t+{best_timesteps})')
    
    # Add text annotation explaining the boundary
    plt.annotate('← Forecasts using some training data | Pure forecasts →', 
                 xy=(forecast_boundary_date, plt.ylim()[1]*0.9),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                 rotation=0, fontsize=10)

plt.title('Effect of Trend Adjustment on Test Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()