import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class ExponentialModel:
    def __init__(self, n_data, y_data):
        self.n_data = n_data
        self.y_data = y_data
        self.r2 = None
        self.mse = None

    #model definition
    def exp_model(self, n):
        return 2**n

    def fit(self):
        # Predicted values
        y_pred = self.exp_model(self.n_data)

        #R-squared
        self.r2 = r2_score(self.y_data, y_pred)

        #MSE
        self.mse = mean_squared_error(self.y_data, y_pred)

        print(f"R-squared: {self.r2:.4f}")
        print(f"MSE: {self.mse:.4f}")
        return self.r2, self.mse


    #New Predictions if needed
    def predict(self, n_fit):
        return self.exp_model(n_fit)

    def plot(self):
        #smooth curve
        n_fit = np.linspace(min(self.n_data), max(self.n_data), 100)  #data range
        y_fit = self.predict(n_fit)

        #model vs fitted
        plt.scatter(self.n_data, self.y_data, label="Data Points", color="blue")
        plt.plot(n_fit, y_fit, label=r"Fitted Model: $2^n$", color="red")
        plt.xlabel("n")
        plt.ylabel("y")
        plt.legend()
        plt.title("Fitting to $2^n$ Model")
        plt.show()

