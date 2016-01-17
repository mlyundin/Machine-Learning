import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from common_functions import add_zero_feature, cf_lr as cost_function, gf_lr as grad_function, \
cf_lr_reg as cost_function_reg, gf_lr_reg as grad_function_reg

if __name__ == '__main__':
    data = sio.loadmat('ex3data1.mat')

    y = data['y']
    X = data['X']

    # replace 10 by 0
    y = y % 10

    n_sampels = 100
    sampels = np.random.choice(len(X), n_sampels)

    fig = plt.figure(figsize=(8, 8))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i, j in enumerate(sampels):
        ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[j, :].reshape(20, 20).T, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(y[j, 0]))

    plt.show()

    num_labels = 10
    X = add_zero_feature(X)
    m, n = X.shape
    initial_theta = np.ones((n, 1))


    all_theta = np.vstack([minimize(cost_function, initial_theta, method='BFGS', jac=grad_function, options={'disp': True, 'maxiter':100},
                    args=(X, (y == i).astype(int))).x for i in range(num_labels)])

    y_pred = np.argmax(np.dot(X, all_theta.T), axis=1)

    print 'Training Set Accuracy: {}'.format(np.mean(y_pred == y.ravel()) * 100)

    # Use regularization
    lambda_coef = 0.1
    all_theta = np.vstack([minimize(cost_function_reg, initial_theta, method='BFGS', jac=grad_function_reg, options={'disp': True, 'maxiter':100},
                                    args=(X, (y == i).astype(int), lambda_coef)).x for i in range(num_labels)])
    y_pred = np.argmax(np.dot(X, all_theta.T), axis=1)

    print 'Training Set Accuracy: {}'.format(np.mean(y_pred == y.ravel()) * 100)