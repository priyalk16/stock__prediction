import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Specify the file path of the processed dataset
file_path = r"C:\Users\Priyal Kantharia\Desktop\Stock Project\processed_mrf_stock_data.csv"

# Load the processed dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit(1)

# Convert 'Date' column to datetime if needed
df['Date'] = pd.to_datetime(df['Date'])

# Display the first few rows of the DataFrame
print(df.head())

# Check summary statistics of numerical columns
print(df.describe())

# Visualize key features (e.g., closing price over time)
plt.figure(figsize=(15, 6))

# Plot closing price against date
plt.plot(df['Date'], df['Close'], label='Closing Price')

# Customize x-axis tick format and rotation
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Set tick interval to every 7 days

plt.title('MRF Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Feature Engineering: Creating lagged features for closing price
df['Close_Lag1'] = df['Close'].shift(1)  # Lagged closing price (previous day)
df['Close_Lag7'] = df['Close'].shift(7)  # Lagged closing price (7 days ago)

# Split the dataset into features (X) and target variable (y)
features = ['Close_Lag1', 'Close_Lag7']
X = df[features]
y = df['Close']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions on the test set using Linear Regression
y_pred_lr = model_lr.predict(X_test)

# Evaluate the Linear Regression model's performance
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

print(f'Linear Regression Mean Absolute Error (MAE): {mae_lr:.2f}')
print(f'Linear Regression Root Mean Squared Error (RMSE): {rmse_lr:.2f}')

# Train a Random Forest Regressor model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions using the Random Forest model
y_pred_rf = model_rf.predict(X_test)

# Evaluate the Random Forest model's performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

print(f'Random Forest Mean Absolute Error (MAE): {mae_rf:.2f}')
print(f'Random Forest Root Mean Squared Error (RMSE): {rmse_rf:.2f}')




# Feature Engineering: Creating lagged features for closing price
df['Close_Lag30'] = df['Close'].shift(30)  # Lagged closing price (30 days ago)
df['Close_Lag90'] = df['Close'].shift(90)  # Lagged closing price (90 days ago)

# Compute moving averages
df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
df['MA_200'] = ta.trend.sma_indicator(df['Close'], window=200)

# Compute RSI (Relative Strength Index)
df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)

# Split the dataset into features (X) and target variable (y) after feature engineering
features = ['Close_Lag1', 'Close_Lag7', 'Close_Lag30', 'Close_Lag90', 'MA_50', 'MA_200', 'RSI_14']
X = df[features]
y = df['Close']


# Evaluate Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

# Evaluate Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

print(f'Linear Regression Mean Absolute Error (MAE): {mae_lr:.2f}')
print(f'Linear Regression Root Mean Squared Error (RMSE): {rmse_lr:.2f}')
print(f'Random Forest Mean Absolute Error (MAE): {mae_rf:.2f}')
print(f'Random Forest Root Mean Squared Error (RMSE): {rmse_rf:.2f}')




# Split the dataset into features (X) and target variable (y) after feature engineering
features = ['Close_Lag1', 'Close_Lag7', 'Close_Lag30', 'Close_Lag90', 'MA_50', 'MA_200', 'RSI_14']
X = df[features]
y = df['Close']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions on the test set using Linear Regression
y_pred_lr = model_lr.predict(X_test)

# Evaluate the Linear Regression model's performance
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
print(f'Linear Regression Mean Absolute Error (MAE): {mae_lr:.2f}')
print(f'Linear Regression Root Mean Squared Error (RMSE): {rmse_lr:.2f}')

# Initialize and train a Random Forest Regressor model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions using the Random Forest model
y_pred_rf = model_rf.predict(X_test)

# Evaluate the Random Forest model's performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f'Random Forest Mean Absolute Error (MAE): {mae_rf:.2f}')
print(f'Random Forest Root Mean Squared Error (RMSE): {rmse_rf:.2f}')

# Plot histograms of actual vs. predicted closing prices
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=30, alpha=0.5, label='Actual Prices', color='blue')
plt.hist(y_pred_rf, bins=30, alpha=0.5, label='Predicted Prices (Random Forest)', color='red')
plt.legend()
plt.show()

plt.title('Distribution of Actual vs. Predicted Closing Prices')
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
