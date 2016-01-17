import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from common_functions import load_data, add_zero_feature, lr_accuracy, cf_lr_norm as cost_function, gf_lr_norm as grad_function

def map_feature(X1, X2, degree=6):
    return add_zero_feature(np.hstack([X1**(i-j)*X2**j for i in range(1, degree+1) for j in range(i+1)]))

if __name__ == '__main__':
    X, y = load_data('ex2data2.txt')

    x1, x2 = X.T
    f_y = y.ravel()
    plt.plot(x1[f_y == 0], x2[f_y == 0], 'yo')
    plt.plot(x1[f_y == 1], x2[f_y == 1], 'bx')

    plt.show()

    X = map_feature(X[:, 0:1], X[:, 1:])
    m, n = X.shape

    initial_theta = np.ones((n, 1))
    lambda_coef = 0.1
    theta = minimize(cost_function, initial_theta, method='BFGS', jac=grad_function, options={'disp': False},
                    args=(X, y, lambda_coef)).x

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    temp = np.array([[i, j] for i in u for j in v])
    temp_X = map_feature(*np.hsplit(temp, 2))
    z = np.dot(temp_X, theta).reshape(len(u), len(v))

    plt.plot(x1[f_y == 0], x2[f_y == 0], 'yo')
    plt.plot(x1[f_y == 1], x2[f_y == 1], 'bx')
    CS = plt.contour(u, v, z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()

    for lambda_coef in (0.03, 0.3, 0.1, 1, 3, 10):
        initial_theta = np.ones((n, 1))
        theta = minimize(cost_function, initial_theta, method='BFGS', jac=grad_function, options={'disp': False},
                        args=(X, y, lambda_coef)).x
        print 'lambda = {}, Train Accuracy: {}'.format(lambda_coef, lr_accuracy(X, y, theta))