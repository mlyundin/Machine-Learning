import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from common_functions import load_data

if __name__ == '__main__':
    X, y = load_data('ex1data2.txt')

    ss = StandardScaler()
    reg = LinearRegression()
    pipeline = Pipeline([("standart_scaler", ss),
                         ("linear_regression", reg)])

    pipeline.fit(X, y)
    prize = pipeline.predict(np.array([1650.0, 3]))
    print prize
