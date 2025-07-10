import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit

class FactorialLogModel:
    def __init__(self, n_data, y_data):
        self.n_data = n_data
        self.y_data = y_data
        self.params = None
        self.r2 = None
        self.mse = None

    # Model Definition (log(n!) = a * n + b)
    def log_factorial_model(self, n, a, b):
        return a * n + b

    def fit(self):
        # Convert y_data to log scale
        log_y_data = np.log(self.y_data, where=(self.y_data > 0))  # Avoid log of 0
        log_y_data = np.nan_to_num(log_y_data, nan=0.0)  # Replace NaN with 0 (if any)
        
        params, _ = curve_fit(self.log_factorial_model, self.n_data, log_y_data, p0=[1, 1])  # Initial guesses for a, b
        self.params = params
        a, b = self.params
        print(f"Fitted parameters: a = {a:.4f}, b = {b:.4f}")

        # Predicted log values
        y_pred_log = self.log_factorial_model(self.n_data, a, b)
        y_pred = np.exp(y_pred_log)  # Convert back to the original scale

        # Calculate R-squared
        self.r2 = r2_score(self.y_data, y_pred)

        # Calculate Mean Squared Error (MSE)
        self.mse = mean_squared_error(self.y_data, y_pred)

        print(f"R-squared: {self.r2:.4f}")
        print(f"MSE: {self.mse:.4f}")

    def predict(self, n_fit):
        if self.params is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        a, b = self.params
        y_pred_log = self.log_factorial_model(n_fit, a, b)
        return np.exp(y_pred_log)  # Convert back to the original scale

    def plot(self):
        # Generate smooth curve for plotting
        n_fit = np.linspace(min(self.n_data), max(self.n_data), 100)  # Smooth curve within data range
        y_fit = self.predict(n_fit)

        # Plot the data and the fitted model
        plt.scatter(self.n_data, self.y_data, label="Data Points", color="blue")
        plt.plot(n_fit, y_fit, label=f"Fitted Model: exp({self.params[0]:.2f} * n + {self.params[1]:.2f})", color="red")
        plt.xlabel("n")
        plt.ylabel("y")
        plt.legend()
        plt.title("Fitting a Factorial Model using Log(n!)")
        plt.show()
