import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize

from common_functions import load_data, add_zero_feature, lr_accuracy, cf_lr_not_norm as cost_function, gf_lr_not_norm as grad_function

if __name__ == '__main__':
    X, y = load_data('ex2data1.txt')

    x1, x2 = X.T
    f_y = y.ravel()
    plt.plot(x1[f_y==0], x2[f_y==0], 'yo')
    plt.plot(x1[f_y==1], x2[f_y==1], 'bx')

    plt.show()

    X = add_zero_feature(X)
    m, n = X.shape
    initial_theta = np.ones((n, 1))

    theta = minimize(cost_function, initial_theta, method='BFGS', jac=grad_function, options={'disp': False},
                    args=(X, y)).x
    print theta
    print cost_function(theta, X, y)

    x1_boundery = np.array([np.min(x1)-2, np.max(x1)+2])
    x2_boundery = (-1/theta[2])*(theta[1]*x1_boundery + theta[0])

    plt.plot(x1[f_y==0], x2[f_y==0], 'yo')
    plt.plot(x1[f_y==1], x2[f_y==1], 'bx')
    plt.plot(x1_boundery, x2_boundery)
    plt.show()


    print 'Train Accuracy: {}'.format(lr_accuracy(X, y, theta))


