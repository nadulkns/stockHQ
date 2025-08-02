import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not available. Using linear regression instead.")
    from sklearn.linear_model import LinearRegression
    KERAS_AVAILABLE = False
    

class StockForecaster:
    def __init__(self, symbol, period="2y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 60

    def fetch_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)

            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            print(f"Fetched {len(self.data)} days of data for {self.symbol}")
            print(
                f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")

            return self.data
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
    def preprocess_data(self, target_column='Close'):
        if self.data is None:
            print("No data available. Please fetch data first.")
            return None, None

        # Use the target column for prediction
        prices = self.data[target_column].values.reshape(-1, 1)

        # Scale the data
        scaled_prices = self.scaler.fit_transform(prices)

        # Create sequences for training
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i, 0])
            y.append(scaled_prices[i, 0])

        X, y = np.array(X), np.array(y)

        # Split data (80% train, 20% test)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if KERAS_AVAILABLE:
            # Reshape for LSTM (samples, time steps, features)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return (X_train, y_train, X_test, y_test)
    
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(50),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_linear_model(self):
        return LinearRegression()
    
    def train_model(self, target_column='Close', epochs=50, batch_size=32):
        # Preprocess data
        data = self.preprocess_data(target_column)
        if data is None:
            return None

        X_train, y_train, X_test, y_test = data

        if KERAS_AVAILABLE:
            # Build and train LSTM model
            self.model = self.build_lstm_model((X_train.shape[1], 1))

            print("Training LSTM model...")
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=0
            )

            # Make predictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

        else:
            # Build and train linear regression model
            self.model = self.build_linear_model()

            print("Training Linear Regression model...")
            self.model.fit(X_train, y_train)

            # Make predictions
            train_pred = self.model.predict(X_train).reshape(-1, 1)
            test_pred = self.model.predict(X_test).reshape(-1, 1)

        # Inverse transform predictions
        train_pred = self.scaler.inverse_transform(train_pred)
        test_pred = self.scaler.inverse_transform(test_pred)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)

        print(f"\nModel Performance:")
        print(f"Training RMSE: ${train_rmse:.2f}")
        print(f"Testing RMSE: ${test_rmse:.2f}")
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Testing MAE: ${test_mae:.2f}")

        return {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'y_train_actual': y_train_actual,
            'y_test_actual': y_test_actual,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
    def predict_future(self, days=30):
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None

        # Get the last sequence_length days of data
        last_sequence = self.data['Close'].tail(self.sequence_length).values
        last_sequence_scaled = self.scaler.transform(
            last_sequence.reshape(-1, 1))

        predictions = []
        current_sequence = last_sequence_scaled.flatten()

        for _ in range(days):
            if KERAS_AVAILABLE:
                # For LSTM
                current_input = current_sequence[-self.sequence_length:].reshape(
                    1, self.sequence_length, 1)
                next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            else:
                # For Linear Regression
                current_input = current_sequence[-self.sequence_length:].reshape(
                    1, -1)
                next_pred = self.model.predict(current_input)[0]

            predictions.append(next_pred)
            current_sequence = np.append(current_sequence, next_pred)

        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        # Create future dates
        last_date = self.data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=days, freq='D')

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions.flatten()
        })

        return future_df
    
    def plot_results(self, results, future_predictions=None):
        plt.figure(figsize=(15, 10))

        # Plot 1: Historical data and predictions
        plt.subplot(2, 1, 1)

        # Historical prices
        plt.plot(self.data.index,
                 self.data['Close'], label='Actual Price', alpha=0.7)

        # Training predictions
        train_start_idx = self.sequence_length
        train_end_idx = train_start_idx + len(results['train_pred'])
        train_dates = self.data.index[train_start_idx:train_end_idx]
        plt.plot(train_dates, results['train_pred'],
                 label='Training Predictions', alpha=0.7)

        # Testing predictions
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + len(results['test_pred'])
        test_dates = self.data.index[test_start_idx:test_end_idx]
        plt.plot(test_dates, results['test_pred'],
                 label='Testing Predictions', alpha=0.7)

        # Future predictions
        if future_predictions is not None:
            plt.plot(future_predictions['Date'], future_predictions['Predicted_Price'],
                     label='Future Predictions', linestyle='--', linewidth=2)

        plt.title(f'{self.symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Prediction accuracy
        plt.subplot(2, 1, 2)
        plt.plot(results['y_test_actual'], label='Actual', alpha=0.7)
        plt.plot(results['test_pred'], label='Predicted', alpha=0.7)
        plt.title('Test Set: Actual vs Predicted')
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
    def get_stock_info(self):
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info

            print(f"\n{self.symbol} Stock Information:")
            print(f"Company: {info.get('longName', 'N/A')}")
            print(f"Sector: {info.get('sector', 'N/A')}")
            print(f"Industry: {info.get('industry', 'N/A')}")
            print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get(
                'marketCap') else "Market Cap: N/A")
            print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
            print(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
            print(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}")

        except Exception as e:
            print(f"Error getting stock info: {e}")