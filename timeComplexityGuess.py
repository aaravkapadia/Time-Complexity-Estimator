from quadReg import QuadraticModel;
from nlognRegression import NLogNModel;
from logreg import LogModel
from linreg import LinearModel
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import timeit
import itertools

#example O(n) algorithim
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
    

