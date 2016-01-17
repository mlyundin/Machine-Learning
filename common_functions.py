import numpy as np

def load_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')

    X = data[:, :-1]
    y = data[:, -1:]

    return X, y

def transform_arguments(tranformation):
    def dec(f):
        def wrapper(*args, **kwargs):
            t_args = map(tranformation, args)
            t_kwargs = {k: tranformation(v) for k, v in kwargs.iteritems()}
            return f(*t_args, **t_kwargs)
        return wrapper
    return dec

matrix_args = transform_arguments(lambda arg: np.matrix(arg, copy=False))
matrix_args_array_only = transform_arguments(lambda arg: np.matrix(arg, copy=False) if isinstance(arg, np.ndarray) else arg)

@matrix_args
def J_liner_regression(X, y, theta):
    temp = X*theta - y

    return (temp.T*temp/(2*len(y)))[0, 0]

@matrix_args_array_only
def gradient_descent(cost_function, X, y, iterations, intial_theta, alpha):
    m = len(y)
    theta = intial_theta
    J_history = []

    for _ in xrange(iterations):
        theta = theta - (alpha/m)*X.T*(X * theta - y)
        J_history.append(cost_function(X, y, theta))

    return theta, J_history

def add_zero_feature(X, axis=1):
    return np.append(np.ones((X.shape[0], 1) if axis else (1, X.shape[1])), X, axis=axis)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def lr_accuracy(X, y, theta):
    theta = theta[:, np.newaxis]
    temp = sigmoid(np.dot(X, theta)).ravel()

    p = np.zeros(len(X))
    p[temp >= 0.5] = 1

    return np.mean(p == y.ravel())*100

@matrix_args
def cf_lr_not_norm(theta, X, y):
    theta = theta.T
    m = len(y)
    Z = sigmoid(X*theta)
    J = (-y.T*np.log(Z) - (1-y).T*np.log(1-Z))/m

    return J[0, 0]

@matrix_args
def gf_lr_not_norm(theta, X, y):
    theta = theta.T
    m = len(y)
    res = (X.T*(sigmoid(X*theta)-y))/m

    return res.A1

@matrix_args_array_only
def cf_lr_norm(theta, X, y, lambda_coef):
    theta = theta.T
    m = len(y)
    Z = sigmoid(X*theta)
    J = (-y.T * np.log(Z) - (1-y).T * np.log(1-Z))/m + (lambda_coef/(2 * m))*theta.T*theta

    return J[0, 0]

@matrix_args_array_only
def gf_lr_norm(theta, X, y, lambda_coef):
    theta = np.matrix(theta.T, copy=True)

    m = len(y)
    Z = X*theta
    theta[0, 0] = 0

    res = (X.T*(sigmoid(Z)-y))/m + (lambda_coef/m)*theta

    return res.A1
