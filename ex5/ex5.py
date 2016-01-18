import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

from common_functions import add_zero_feature, matrix_args_array_only, feature_normalize

@matrix_args_array_only
def cost_function(theta, X, y, lambda_coef):
    theta = np.matrix(theta, copy=True).T
    delta = X*theta-y
    theta[0, 0] = 0
    m = len(y)

    return (1.0/(2*m)*delta.T*delta+float(lambda_coef)/(2*m)*theta.T*theta)[0, 0]

@matrix_args_array_only
def grad_function(theta, X, y, lambda_coef):
    theta = np.matrix(theta, copy=True).T
    delta = X*theta-y
    theta[0, 0] = 0
    m = len(y)

    return (X.T*delta/m + float(lambda_coef)/m*theta).A1

def train_linear_regression(X, y, lambda_coef):
    initial_theta = np.zeros(X.shape[1])
    return minimize(cost_function, initial_theta, method='BFGS', jac=grad_function, options={'disp': False},
                    args=(X, y, lambda_coef)).x

def learning_curve(X, y, Xval, yval, lambda_coef):
    m = len(X)
    error_train = []
    error_val = []
    for i in range(1, m+1):
        I_X = X[:i, :]
        I_y = y[:i]

        theta = train_linear_regression(I_X, I_y, lambda_coef)
        error_train.append(cost_function(theta, I_X, I_y, 0))
        error_val.append(cost_function(theta, Xval, yval, 0))

    return error_train, error_val

def poly_features(X, p):
    return np.hstack([X**i for i in xrange(1, p+1)])

if __name__ == '__main__':
    data = sio.loadmat('ex5data1.mat')

    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']
    m = len(X)

    def plot_data():
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')

        plt.plot(X, y, 'rx')

    plot_data()
    plt.show()

    X_extended = add_zero_feature(X)
    theta = np.array([1, 1])

    print 'J = {}, gradient = {}'.format(cost_function(theta, X_extended, y, 1), grad_function(theta, X_extended, y, 1))

    theta = train_linear_regression(X_extended, y, 1)
    plot_data()
    plt.plot(X, np.dot(X_extended, theta[:, np.newaxis]).ravel())
    plt.show()

    lambda_coef = 0
    error_train, error_val = learning_curve(X_extended, y, add_zero_feature(Xval), yval, lambda_coef)

    plt.plot(range(1, m+1), error_train, label='Train')
    plt.plot(range(1, m+1), error_val, c='r', label='Cross validation')

    plt.legend()
    plt.title('Learning curve for linear regression (pow of polynomial = 1)')
    plt.xlabel('Number of training examples)')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()

    p = 8
    mu, sigma, X_poly = feature_normalize(poly_features(X, p))
    X_poly = add_zero_feature(X_poly)

    prepare_X = lambda X: add_zero_feature((poly_features(X, p)-m)/sigma)

    X_poly_test = prepare_X(Xtest)
    X_poly_val = prepare_X(Xval)

    lambda_coef = 1
    theta = train_linear_regression(X_poly, y, lambda_coef)

    x = np.arange(np.min(X) - 15, np.max(X) + 25, 0.05)[:, np.newaxis]
    x_poly = prepare_X(x)

    plt.plot(x, np.dot(x_poly, theta))
    plot_data()
    plt.show()

    error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, lambda_coef)

    plt.title('Learning curve for linear regression (lambda = {}, pow of polynomial = {} )'.format(lambda_coef, p))
    plt.plot(range(1, m+1), error_train, label='Train')
    plt.plot(range(1, m+1), error_val, c='r', label='Cross validation')
    plt.legend()
    plt.axis([0, 13, 0, 50])
    plt.show()
