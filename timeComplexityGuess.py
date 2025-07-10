from quadReg import QuadraticModel;
from expReg import ExponentialModel;
from nlognRegression import NLogNModel;
from factReg import FactorialLogModel
from logreg import LogModel
from linreg import LinearModel
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import timeit
import itertools


# n_data = np.array([1, 5, 10, 50, 100, 500, 1000, 5000, 10000])
# y_data = np.array([0, 8, 11, 19, 23, 31, 34, 42, 46])

# # Create an instance of the LogModel
# model = NLogNModel(n_data, y_data)

# # Fit the model to the data
# model.fit()

# # Optionally plot the results
# model.plot()

#1. create function that takes in a function as an arguement.
    #function must run the argument function and measure how long it takes for the function to run
#2. create second function that takes in function number 1 as an argument.
    #Function runs a while loop for inputs 1, 10, 100, 1000, and 10000, adds the amount of time into a np array and returns the array
#3. Final function takes in np array as argument
    #function will run np array through all regressions, and get the r-squared value of each regression, in order to find the most likely time complexity

# def measureTime(algo, *args):
#     time = timeit.timeit(lambda: algo(*args), number=1) * 1000  
#     return time

def algo(n):
    s = 0
    for i in range(n):
        s += i
    return s

def measureTime(algo, n):
    time = timeit.timeit(lambda: algo(n), number=1) * 1000  
    return time

def createArray(algo, n_values):
    times = []
    for n in n_values:
        times.append(measureTime(algo, n))
    return np.array(times) 

def logTest(n_data, y_data):
    model = LogModel(n_data, y_data)
    r2, mse = model.fit()
    model.plot()
    return r2

def linTest(n_data, y_data):
    model = LinearModel(n_data, y_data)
    r2, mse = model.fit()
    model.plot()
    return r2

def linlogTest(n_data, y_data):
    model = NLogNModel(n_data, y_data)
    r2, mse = model.fit()
    model.plot()
    return r2

def quadTest(n_data, y_data):
    model = QuadraticModel(n_data, y_data)
    r2, mse = model.fit()
    model.plot()
    return r2

def expTest(n_data, y_data):
    model = ExponentialModel(n_data, y_data)
    r2, mse = model.fit()
    model.plot()
    return r2

def compare(algo):
    n_data = np.array([1, 5, 10, 50, 100, 500, 1000, 5000, 10000])
    y_data = createArray(algo, n_data)
    models = [logTest, linTest, linlogTest, quadTest]
    modelstrings = ["log time", "linear time", "nlogn time", "quadratic time"]
    i = 0
    max = 0
    index = 0
    while(i <= 3):
        r2 = models[i](n_data, y_data)
        if r2 > max:
            max = r2
            index = i
        i += 1
    return modelstrings[index], max
            
    
print(compare(algo))
    

