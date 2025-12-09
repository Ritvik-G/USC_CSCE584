"""
Agent 1: Price Forecasting
Simple neural network to predict electricity prices from day + hour
Uses scikit-learn MLPRegressor (no TensorFlow, no lock errors)
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

# Set publication-quality plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
sns.set_palette("husl")


class PriceForecastingAgent:
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metrics = {}
        self.training_history = []
        
        # Store predictions for plotting
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_pred = None
        self.y_val_pred = None
        self.y_test_pred = None
        
    def create_features(self, day, hour):
        """Create cyclical time features"""
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 365)
        day_cos = np.cos(2 * np.pi * day / 365)
        
        return np.column_stack([hour_sin, hour_cos, day_sin, day_cos, hour/24, day/365])
    
    def train(self, filepath, max_iter=500):
        """Train the model"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        df['Price_kWh'] = df['Electricity Price [$/MWh]'] / 1000
        
        # Prepare features
        X = self.create_features(df['Day'].values, df['Hour'].values)
        y = df['Price_kWh'].values
        
        # Split data: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Scale
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Build and train model
        print(f"Training (architecture: 6 → 64 → 32 → 16 → 1)...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=42,
            verbose=True,
            early_stopping=False
        )
        
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Store training loss curve
        self.training_history = self.model.loss_curve_
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_val_pred_scaled = self.model.predict(X_val_scaled)
        y_test_pred_scaled = self.model.predict(X_test_scaled)
        
        # Inverse transform
        self.y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        self.y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
        self.y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        
        # Store actual values
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
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Train - R²: {self.metrics['train']['r2']:.4f}, MAE: ${self.metrics['train']['mae']:.6f}/kWh, RMSE: ${self.metrics['train']['rmse']:.6f}/kWh")
        print(f"Val   - R²: {self.metrics['val']['r2']:.4f}, MAE: ${self.metrics['val']['mae']:.6f}/kWh, RMSE: ${self.metrics['val']['rmse']:.6f}/kWh")
        print(f"Test  - R²: {self.metrics['test']['r2']:.4f}, MAE: ${self.metrics['test']['mae']:.6f}/kWh, RMSE: ${self.metrics['test']['rmse']:.6f}/kWh")
        print(f"{'='*60}")
        
    def plot_results(self, save_path='agent1_results.png'):
        """Generate publication-quality plots"""
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Training Loss Curve
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(self.training_history, linewidth=2, color='#2E86AB')
        plt.xlabel('Iteration', fontweight='bold')
        plt.ylabel('Loss (MSE)', fontweight='bold')
        plt.title('Training Loss Curve', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 2. Predicted vs Actual - Train
        ax2 = plt.subplot(2, 3, 2)
        plt.scatter(self.y_train, self.y_train_pred, alpha=0.5, s=10, color='#A23B72')
        plt.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 
                'k--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($/kWh)', fontweight='bold')
        plt.ylabel('Predicted Price ($/kWh)', fontweight='bold')
        plt.title(f'Train Set (R²={self.metrics["train"]["r2"]:.4f})', fontweight='bold', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Predicted vs Actual - Validation
        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(self.y_val, self.y_val_pred, alpha=0.5, s=10, color='#F18F01')
        plt.plot([self.y_val.min(), self.y_val.max()], 
                [self.y_val.min(), self.y_val.max()], 
                'k--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($/kWh)', fontweight='bold')
        plt.ylabel('Predicted Price ($/kWh)', fontweight='bold')
        plt.title(f'Validation Set (R²={self.metrics["val"]["r2"]:.4f})', fontweight='bold', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Predicted vs Actual - Test
        ax4 = plt.subplot(2, 3, 4)
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, s=15, color='#C73E1D')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'k--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($/kWh)', fontweight='bold')
        plt.ylabel('Predicted Price ($/kWh)', fontweight='bold')
        plt.title(f'Test Set (R²={self.metrics["test"]["r2"]:.4f})', fontweight='bold', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Residual Plot - Test Set
        ax5 = plt.subplot(2, 3, 5)
        residuals = self.y_test - self.y_test_pred
        plt.scatter(self.y_test_pred, residuals, alpha=0.6, s=15, color='#6A4C93')
        plt.axhline(y=0, color='k', linestyle='--', lw=2)
        plt.xlabel('Predicted Price ($/kWh)', fontweight='bold')
        plt.ylabel('Residuals ($/kWh)', fontweight='bold')
        plt.title('Residual Plot (Test Set)', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 6. Metrics Comparison Bar Chart
        ax6 = plt.subplot(2, 3, 6)
        metrics_data = {
            'Train': [self.metrics['train']['r2'], self.metrics['train']['mae']*100, self.metrics['train']['rmse']*100],
            'Val': [self.metrics['val']['r2'], self.metrics['val']['mae']*100, self.metrics['val']['rmse']*100],
            'Test': [self.metrics['test']['r2'], self.metrics['test']['mae']*100, self.metrics['test']['rmse']*100]
        }
        
        x = np.arange(3)
        width = 0.25
        
        plt.bar(x - width, metrics_data['Train'], width, label='Train', color='#A23B72')
        plt.bar(x, metrics_data['Val'], width, label='Val', color='#F18F01')
        plt.bar(x + width, metrics_data['Test'], width, label='Test', color='#C73E1D')
        
        plt.xlabel('Metrics', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        plt.title('Performance Metrics Comparison', fontweight='bold', fontsize=12)
        plt.xticks(x, ['R²', 'MAE (¢/kWh)', 'RMSE (¢/kWh)'])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Agent 1: Electricity Price Forecasting - Model Evaluation', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved: {save_path}")
        plt.show()
        
    def generate_report(self, save_path='agent1_report.txt'):
        """Generate text report for paper"""
        
        report = f"""
{'='*70}
AGENT 1: ELECTRICITY PRICE FORECASTING MODEL - EVALUATION REPORT
{'='*70}

MODEL ARCHITECTURE:
- Input Features: 6 (cyclical encoding of day and hour)
- Hidden Layers: 64 → 32 → 16 neurons (ReLU activation)
- Output: 1 (electricity price in $/kWh)
- Optimizer: Adam
- Training Iterations: {len(self.training_history)}

DATA SPLIT:
- Training Set: {len(self.y_train)} samples (70%)
- Validation Set: {len(self.y_val)} samples (15%)
- Test Set: {len(self.y_test)} samples (15%)

PERFORMANCE METRICS:

Training Set:
  - R² Score: {self.metrics['train']['r2']:.4f}
  - MAE: ${self.metrics['train']['mae']:.6f}/kWh
  - RMSE: ${self.metrics['train']['rmse']:.6f}/kWh
  - MAPE: {self.metrics['train']['mape']:.2f}%

Validation Set:
  - R² Score: {self.metrics['val']['r2']:.4f}
  - MAE: ${self.metrics['val']['mae']:.6f}/kWh
  - RMSE: ${self.metrics['val']['rmse']:.6f}/kWh
  - MAPE: {self.metrics['val']['mape']:.2f}%

Test Set (Unseen Data):
  - R² Score: {self.metrics['test']['r2']:.4f}
  - MAE: ${self.metrics['test']['mae']:.6f}/kWh
  - RMSE: ${self.metrics['test']['rmse']:.6f}/kWh
  - MAPE: {self.metrics['test']['mape']:.2f}%

INTERPRETATION:
The model achieves an R² of {self.metrics['test']['r2']:.4f} on the test set, 
explaining {self.metrics['test']['r2']*100:.2f}% of the variance in electricity prices.
Predictions are typically within ${self.metrics['test']['mae']:.6f}/kWh 
({self.metrics['test']['mape']:.2f}% MAPE) of actual values.

The close alignment between training, validation, and test metrics 
(R² difference: {abs(self.metrics['train']['r2'] - self.metrics['test']['r2']):.4f})
indicates good generalization with minimal overfitting.

{'='*70}
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Report saved: {save_path}")
        
    def predict(self, day, hour):
        """Predict price for given day and hour"""
        X = self.create_features(np.array([day]), np.array([hour]))
        X_scaled = self.scaler_X.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        price = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return max(0, price)
    
    def predict_day(self, day):
        """Predict all 24 hours for a day"""
        hours = np.arange(1, 25)
        days = np.full(24, day)
        X = self.create_features(days, hours)
        X_scaled = self.scaler_X.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        prices = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        return {int(h): max(0, float(p)) for h, p in zip(hours, prices)}
    
    def save(self, prefix='agent1'):
        """Save model"""
        with open(f'{prefix}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{prefix}_scalers.pkl', 'wb') as f:
            pickle.dump({'X': self.scaler_X, 'y': self.scaler_y}, f)
        with open(f'{prefix}_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nSaved: {prefix}_model.pkl, {prefix}_scalers.pkl, {prefix}_metrics.json")
    
    def load(self, prefix='agent1'):
        """Load model"""
        with open(f'{prefix}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{prefix}_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_X = scalers['X']
            self.scaler_y = scalers['y']
        with open(f'{prefix}_metrics.json', 'r') as f:
            self.metrics = json.load(f)
        print(f"Loaded: {prefix}_model.pkl")


if __name__ == "__main__":
    # Train
    agent = PriceForecastingAgent()
    agent.train('/Users/ritvikg/Desktop/USC/Classes/CS584/Project/Data/price_forecasting.csv', max_iter=500)
    
    # Generate visualizations
    agent.plot_results('agent1_results.png')
    
    # Generate text report
    agent.generate_report('agent1_report.txt')
    
    # Test predictions
    print(f"\nSample predictions:")
    print(f"Day 180, Hour 14: ${agent.predict(180, 14):.6f}/kWh")
    print(f"Day 1, Hour 1: ${agent.predict(1, 1):.6f}/kWh")
    
    # Save
    agent.save('agent1')

    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metrics = {}
        
    def create_features(self, day, hour):
        """Create cyclical time features"""
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 365)
        day_cos = np.cos(2 * np.pi * day / 365)
        
        return np.column_stack([hour_sin, hour_cos, day_sin, day_cos, hour/24, day/365])
    
    def train(self, filepath, max_iter=500):
        """Train the model"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        df['Price_kWh'] = df['Electricity Price [$/MWh]'] / 1000
        
        # Prepare features
        X = self.create_features(df['Day'].values, df['Hour'].values)
        y = df['Price_kWh'].values
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
        
        # Scale
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)
        
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        
        # Build and train model
        print(f"Training (architecture: 6 → 64 → 32 → 16 → 1)...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=42,
            verbose=True
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.scaler_y.inverse_transform(self.model.predict(X_train).reshape(-1, 1)).ravel()
        y_val_pred = self.scaler_y.inverse_transform(self.model.predict(X_val).reshape(-1, 1)).ravel()
        y_test_pred = self.scaler_y.inverse_transform(self.model.predict(X_test).reshape(-1, 1)).ravel()
        
        y_train_true = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        
        self.metrics = {
            'train': {
                'r2': r2_score(y_train_true, y_train_pred),
                'mae': mean_absolute_error(y_train_true, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred))
            },
            'val': {
                'r2': r2_score(y_val, y_val_pred),
                'mae': mean_absolute_error(y_val, y_val_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred))
            },
            'test': {
                'r2': r2_score(y_test, y_test_pred),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
            }
        }
        
        print(f"\nResults:")
        print(f"Train R²: {self.metrics['train']['r2']:.4f}, MAE: ${self.metrics['train']['mae']:.6f}/kWh")
        print(f"Val   R²: {self.metrics['val']['r2']:.4f}, MAE: ${self.metrics['val']['mae']:.6f}/kWh")
        print(f"Test  R²: {self.metrics['test']['r2']:.4f}, MAE: ${self.metrics['test']['mae']:.6f}/kWh")
        
    def predict(self, day, hour):
        """Predict price for given day and hour"""
        X = self.create_features(np.array([day]), np.array([hour]))
        X_scaled = self.scaler_X.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        price = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return max(0, price)
    
    def predict_day(self, day):
        """Predict all 24 hours for a day"""
        hours = np.arange(1, 25)
        days = np.full(24, day)
        X = self.create_features(days, hours)
        X_scaled = self.scaler_X.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        prices = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        return {int(h): max(0, float(p)) for h, p in zip(hours, prices)}
    
    def save(self, prefix='agent1'):
        """Save model"""
        with open(f'{prefix}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{prefix}_scalers.pkl', 'wb') as f:
            pickle.dump({'X': self.scaler_X, 'y': self.scaler_y}, f)
        with open(f'{prefix}_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nSaved: {prefix}_model.pkl, {prefix}_scalers.pkl, {prefix}_metrics.json")
    
    def load(self, prefix='agent1'):
        """Load model"""
        with open(f'{prefix}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{prefix}_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_X = scalers['X']
            self.scaler_y = scalers['y']
        with open(f'{prefix}_metrics.json', 'r') as f:
            self.metrics = json.load(f)
        print(f"Loaded: {prefix}_model.pkl")


if __name__ == "__main__":
    # Train
    agent = PriceForecastingAgent()
    agent.train('/Users/ritvikg/Desktop/USC/Classes/CS584/Project/Data/price_forecasting.csv', max_iter=500)
    
    # Test predictions
    print(f"\nSample predictions:")
    print(f"Day 180, Hour 14: ${agent.predict(180, 14):.6f}/kWh")
    print(f"Day 1, Hour 1: ${agent.predict(1, 1):.6f}/kWh")
    
    # Save
    agent.save('agent1')