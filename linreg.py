import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

class LinearModel:
    def __init__(self, n_data, y_data):
        self.n_data = n_data
        self.y_data = y_data
        self.r2 = None
        self.mse = None
        self.a = None
        self.b = None

    def lin_model(self, n, a, b):
        return a * n + b

    def fit(self):
        params, _ = curve_fit(self.lin_model, self.n_data, self.y_data, p0=[1, 1])
        self.a, self.b = params 
        print(f"Fitted parameters: a = {self.a:.4f}, b = {self.b:.4f}")
        y_pred = self.lin_model(self.n_data, self.a, self.b)
        self.r2 = r2_score(self.y_data, y_pred)
        self.mse = mean_squared_error(self.y_data, y_pred)
        print(f"R-squared: {self.r2:.4f}")
        print(f"MSE: {self.mse:.4f}")
        return self.r2, self.mse
    
    def predict(self, n_fit):
        return self.lin_model(n_fit, self.a, self.b)
    
    def plot(self):
        n_fit = np.linspace(min(self.n_data), max(self.n_data), 100) 
        y_fit = self.predict(n_fit)
        plt.scatter(self.n_data, self.y_data, label="Data Points", color="blue")
        plt.plot(n_fit, y_fit, label=f"Fitted Model: {self.a:.2f} * n + {self.b:.2f}", color="red")
        plt.xlabel("n")
        plt.ylabel("y")
        plt.legend()
        plt.title("Fitting to a linear Model")
        plt.show()


