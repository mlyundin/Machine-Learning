import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from common_functions import load_data

if __name__ == '__main__':
    X, y = load_data('ex2data2.txt')

    x1, x2 = X.T
    f_y = y.ravel()
    plt.plot(x1[f_y == 0], x2[f_y == 0], 'yo')
    plt.plot(x1[f_y == 1], x2[f_y == 1], 'bx')

    plt.show()

    pf = PolynomialFeatures(degree=6)
    reg = LogisticRegression(C=10)
    pipeline = Pipeline([("polynomial_features", pf),
                         ("logistic_regression", reg)])

    pipeline.fit(X, f_y)
    theta = reg.coef_.T

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    X1, X2 = np.meshgrid(u, v)
    X1, X2 = X1.reshape(-1, 1), X2.reshape(-1, 1)
    temp_X = pf.transform(np.hstack((X1, X2)))

    z = np.dot(temp_X, theta).reshape(len(u), len(v))

    plt.plot(x1[f_y == 0], x2[f_y == 0], 'yo')
    plt.plot(x1[f_y == 1], x2[f_y == 1], 'bx')
    CS = plt.contour(u, v, z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()

    for lambda_coef in (0.03, 0.3, 0.1, 1, 3, 10):
        c = 1.0/lambda_coef
        pipeline.set_params(**{'logistic_regression__C': c})
        pipeline.fit(X, y.ravel())
        print 'lambda = {}, Train Accuracy: {}'.format(lambda_coef, pipeline.score(X, y)*100)
