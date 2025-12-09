"""
Agent 2: Household Power Forecasting
Predicts household power consumption (kW) from day and hour using feed-forward neural network
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


class PowerForecastingAgent:
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metrics = {}
        self.training_history = []
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_pred = None
        self.y_val_pred = None
        self.y_test_pred = None
        
    def create_features(self, day, hour):
        """Create cyclical time features with seasonal encoding"""
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 365)
        day_cos = np.cos(2 * np.pi * day / 365)
        
        # Month encoding for seasonal patterns
        month = np.ceil(day / 30.5).astype(int)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        return np.column_stack([hour_sin, hour_cos, day_sin, day_cos, 
                               month_sin, month_cos, hour/24, day/365])
    
    def load_and_aggregate_data(self, filepath):
        """Load minute-level data and aggregate to hourly"""
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                       format='%d/%m/%y %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        
        # Extract day and hour
        df['DayOfYear'] = df['DateTime'].dt.dayofyear
        df['Hour'] = df['DateTime'].dt.hour + 1
        
        # Convert to numeric and clean
        numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Global_active_power'])
        
        # Aggregate to hourly
        hourly_df = df.groupby(['DayOfYear', 'Hour']).agg({
            'Global_active_power': 'mean',
            'Global_reactive_power': 'mean',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        }).reset_index()
        
        hourly_df = hourly_df.rename(columns={
            'DayOfYear': 'Day',
            'Global_active_power': 'Power_kW'
        })
        
        return hourly_df
    
    def train(self, filepath, max_iter=500):
        """Train power forecasting model"""
        # Load and aggregate data
        df = self.load_and_aggregate_data(filepath)
        
        X = self.create_features(df['Day'].values, df['Hour'].values)
        y = df['Power_kW'].values
        
        # Split: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
        
        # Normalize
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train neural network: 8 → 64 → 32 → 16 → 1
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=42,
            verbose=True
        )
        self.model.fit(X_train_scaled, y_train_scaled)
        self.training_history = self.model.loss_curve_
        
        # Make predictions
        self.y_train_pred = self.scaler_y.inverse_transform(
            self.model.predict(X_train_scaled).reshape(-1, 1)).ravel()
        self.y_val_pred = self.scaler_y.inverse_transform(
            self.model.predict(X_val_scaled).reshape(-1, 1)).ravel()
        self.y_test_pred = self.scaler_y.inverse_transform(
            self.model.predict(X_test_scaled).reshape(-1, 1)).ravel()
        
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Calculate metrics
        self.metrics = {
            'train': {
                'r2': r2_score(self.y_train, self.y_train_pred),
                'mae': mean_absolute_error(self.y_train, self.y_train_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_train, self.y_train_pred)),
                'mape': np.mean(np.abs((self.y_train - self.y_train_pred) / self.y_train)) * 100
            },
            'val': {
                'r2': r2_score(self.y_val, self.y_val_pred),
                'mae': mean_absolute_error(self.y_val, self.y_val_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_val, self.y_val_pred)),
                'mape': np.mean(np.abs((self.y_val - self.y_val_pred) / self.y_val)) * 100
            },
            'test': {
                'r2': r2_score(self.y_test, self.y_test_pred),
                'mae': mean_absolute_error(self.y_test, self.y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_test_pred)),
                'mape': np.mean(np.abs((self.y_test - self.y_test_pred) / self.y_test)) * 100
            }
        }
        
        print(f"\nTrain R²: {self.metrics['train']['r2']:.4f}")
        print(f"Val   R²: {self.metrics['val']['r2']:.4f}")
        print(f"Test  R²: {self.metrics['test']['r2']:.4f}")
        
    def plot_results(self, save_path='agent2_results.png'):
        """Generate evaluation plots"""
        fig = plt.figure(figsize=(16, 10))
        
        # Training loss
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history, linewidth=2, color='#2E86AB')
        plt.xlabel('Iteration', fontweight='bold')
        plt.ylabel('Loss (MSE)', fontweight='bold')
        plt.title('Training Loss', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Train predictions
        plt.subplot(2, 3, 2)
        plt.scatter(self.y_train, self.y_train_pred, alpha=0.5, s=10, color='#A23B72')
        plt.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 'k--', lw=2)
        plt.xlabel('Actual (kW)', fontweight='bold')
        plt.ylabel('Predicted (kW)', fontweight='bold')
        plt.title(f'Train (R²={self.metrics["train"]["r2"]:.4f})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Validation predictions
        plt.subplot(2, 3, 3)
        plt.scatter(self.y_val, self.y_val_pred, alpha=0.5, s=10, color='#F18F01')
        plt.plot([self.y_val.min(), self.y_val.max()], 
                [self.y_val.min(), self.y_val.max()], 'k--', lw=2)
        plt.xlabel('Actual (kW)', fontweight='bold')
        plt.ylabel('Predicted (kW)', fontweight='bold')
        plt.title(f'Validation (R²={self.metrics["val"]["r2"]:.4f})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Test predictions
        plt.subplot(2, 3, 4)
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, s=15, color='#C73E1D')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual (kW)', fontweight='bold')
        plt.ylabel('Predicted (kW)', fontweight='bold')
        plt.title(f'Test (R²={self.metrics["test"]["r2"]:.4f})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Residuals
        plt.subplot(2, 3, 5)
        residuals = self.y_test - self.y_test_pred
        plt.scatter(self.y_test_pred, residuals, alpha=0.6, s=15, color='#6A4C93')
        plt.axhline(y=0, color='k', linestyle='--', lw=2)
        plt.xlabel('Predicted (kW)', fontweight='bold')
        plt.ylabel('Residuals (kW)', fontweight='bold')
        plt.title('Residual Plot', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Metrics comparison
        plt.subplot(2, 3, 6)
        x = np.arange(3)
        width = 0.25
        plt.bar(x - width, [self.metrics['train']['r2'], 
                self.metrics['train']['mae'], self.metrics['train']['rmse']], 
                width, label='Train', color='#A23B72')
        plt.bar(x, [self.metrics['val']['r2'], 
                self.metrics['val']['mae'], self.metrics['val']['rmse']], 
                width, label='Val', color='#F18F01')
        plt.bar(x + width, [self.metrics['test']['r2'], 
                self.metrics['test']['mae'], self.metrics['test']['rmse']], 
                width, label='Test', color='#C73E1D')
        plt.xlabel('Metrics', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        plt.title('Performance Metrics', fontweight='bold')
        plt.xticks(x, ['R²', 'MAE (kW)', 'RMSE (kW)'])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Agent 2: Power Forecasting', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
    def generate_report(self, save_path='agent2_report.txt'):
        """Generate text report"""
        report = f"""
AGENT 2: POWER FORECASTING

Training Set:
  R²: {self.metrics['train']['r2']:.4f}
  MAE: {self.metrics['train']['mae']:.3f} kW
  RMSE: {self.metrics['train']['rmse']:.3f} kW

Validation Set:
  R²: {self.metrics['val']['r2']:.4f}
  MAE: {self.metrics['val']['mae']:.3f} kW
  RMSE: {self.metrics['val']['rmse']:.3f} kW

Test Set:
  R²: {self.metrics['test']['r2']:.4f}
  MAE: {self.metrics['test']['mae']:.3f} kW
  RMSE: {self.metrics['test']['rmse']:.3f} kW
"""
        with open(save_path, 'w') as f:
            f.write(report)
        print(report)
        
    def predict(self, day, hour):
        """Predict power for single hour"""
        if day < 1 or day > 365:
            print(f"Warning: Day {day} outside typical range")
        if hour < 1 or hour > 24:
            raise ValueError(f"Hour must be 1-24, got {hour}")
            
        X = self.create_features(np.array([day]), np.array([hour]))
        X_scaled = self.scaler_X.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        power = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return max(0, power)
    
    def predict_day(self, day):
        """Predict power for 24 hours"""
        hours = np.arange(1, 25)
        days = np.full(24, day)
        X = self.create_features(days, hours)
        X_scaled = self.scaler_X.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        power_values = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        return {int(h): max(0, float(p)) for h, p in zip(hours, power_values)}
    
    def save(self, prefix='agent2'):
        """Save model and metrics"""
        with open(f'{prefix}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{prefix}_scalers.pkl', 'wb') as f:
            pickle.dump({'X': self.scaler_X, 'y': self.scaler_y}, f)
        with open(f'{prefix}_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, prefix='agent2'):
        """Load model and metrics"""
        with open(f'{prefix}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{prefix}_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_X = scalers['X']
            self.scaler_y = scalers['y']
        with open(f'{prefix}_metrics.json', 'r') as f:
            self.metrics = json.load(f)


if __name__ == "__main__":
    agent = PowerForecastingAgent()
    agent.train('../Data/power_forecasting.csv', max_iter=500)
    agent.plot_results('agent2_results.png')
    agent.generate_report('agent2_report.txt')
    agent.save('agent2')