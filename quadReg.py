import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

class QuadraticModel:
    def __init__(self, n_data, y_data):
        self.n_data = n_data
        self.y_data = y_data
        self.params = None
        self.r2 = None
        self.mse = None

    ##Model Definition
    def quad_model(self, n, a, b):
        return a * n**2 + b

    def fit(self):
        #sci-pi curve fit
        params, covariance = curve_fit(self.quad_model, self.n_data, self.y_data, p0=[1, 1])  # Initial guesses for a, b
        self.params = params
        a, b = self.params
        print(f"Fitted parameters: a = {a:.4f}, b = {b:.4f}")

        #Predicted values
        y_pred = self.quad_model(self.n_data, a, b)

        #R-squared
        self.r2 = r2_score(self.y_data, y_pred)
        #MSE
        self.mse = mean_squared_error(self.y_data, y_pred)

        print(f"R-squared: {self.r2:.4f}")
        print(f"MSE: {self.mse:.4f}")
        return self.r2, self.mse


    #Predicted value(if needed)
    def predict(self, n_fit):
        if self.params is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        a, b = self.params
        return self.quad_model(n_fit, a, b)

    def plot(self):
        #smooth curve
        n_fit = np.linspace(min(self.n_data), max(self.n_data), 100)  #data range
        y_fit = self.predict(n_fit)

        #data vs model
        plt.scatter(self.n_data, self.y_data, label="Data Points", color="blue")
        plt.plot(n_fit, y_fit, label=f"Fitted Model: {self.params[0]:.2f} * n^2 + {self.params[1]:.2f}", color="red")
        plt.xlabel("n")
        plt.ylabel("y")
        plt.legend()
        plt.title("Fitting a n^2 Model")
        plt.show()
