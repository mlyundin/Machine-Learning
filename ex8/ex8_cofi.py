import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import norm

from scipy.optimize import minimize
import scipy.io as sio

def cofi_J(params, Y, R, num_users, num_movies, num_features, lambda_coef):
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    temp = (X.dot(Theta.T) - Y)*R
    return np.sum(temp**2)/2 + lambda_coef/2*np.sum(X**2) + lambda_coef/2*np.sum(Theta**2)

def cofi_grad(params, Y, R, num_users, num_movies, num_features, lambda_coef):
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    temp = (X.dot(Theta.T) - Y)*R

    X_grad = temp.dot(Theta)+ lambda_coef*X
    Theta_grad = temp.T.dot(X) + lambda_coef*Theta

    return np.hstack((X_grad.ravel(), Theta_grad.ravel()))

def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, lambda_coef):
    return cofi_J(params, Y, R, num_users, num_movies, num_features, lambda_coef), cofi_grad(params, Y, R, num_users, num_movies, num_features, lambda_coef)

def check_cost_function(lambda_coef=0):
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    Y = X_t.dot(Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0

    R = np.zeros_like(Y)
    R[Y != 0] = 1

    X = np.random.randn(*X_t.shape);
    Theta = np.random.randn(*Theta_t.shape);
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    J = lambda t: cofi_cost_func(t, Y, R, num_users, num_movies, num_features, lambda_coef)[0]
    numgrad = compute_numerical_gradient(J, np.hstack((X.ravel(), Theta.ravel())))

    cost, grad = cofi_cost_func(np.hstack((X.ravel(), Theta.ravel())),  Y, R, num_users, num_movies, num_features, lambda_coef)
    for i, j in zip(numgrad, grad):
        print i, j
    print('The above two columns you get should be very similar.\n'
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n'
         'the relative difference will be small (less than 1e-9). \n'
         'Relative Difference: %s' % diff)

def compute_numerical_gradient(J, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    e = 1e-4

    for p in range(len(theta)):

        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad

def load_movie_list():
    with open('movie_ids.txt') as f:
        return [' '.join(i.split()[1:]) for i in f]

def normalize_ratings(Y, R):
    m = len(Y)
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros_like(Y)

    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i,:] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean

if __name__ == '__main__':
    data = sio.loadmat('ex8_movies.mat')
    Y = data['Y']
    R = data['R'].astype(bool)

    print 'Average rating for movie 1 (Toy Story): %s / 5' % np.mean(Y[0, R[0]])

    imgplot = plt.imshow(Y)
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.show()

    data = sio.loadmat('ex8_movieParams.mat')
    X = data['X']
    Theta = data['Theta']

    num_users = 4
    num_movies = 5
    num_features = 3

    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]
    params = np.hstack((X.ravel(), Theta.ravel()))

    J,_ = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, 0)

    print('Cost at loaded parameters: %s (this value should be about 22.22)' % J)

    print('Checking Gradients (without regularization) ...')
    check_cost_function()

    J, _ = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, 1.5)
    print('Cost at loaded parameters (lambda = 1.5): %s (this value should be about 31.34)' % J)

    print('Checking Gradients (with regularization) ...')
    check_cost_function(1.5)


    movie_list = load_movie_list()
    my_ratings = np.zeros((1682, 1))

    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11]= 5
    my_ratings[53] = 4
    my_ratings[63]= 5
    my_ratings[65]= 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354]= 5

    data = sio.loadmat('ex8_movies.mat')
    Y = data['Y']
    R = data['R'].astype(bool)

    Y = np.hstack((my_ratings, Y))
    R = np.hstack((my_ratings != 0, R))

    Ynorm, Ymean = normalize_ratings(Y, R)

    num_movies, num_users = Y.shape
    num_features = 10

    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)

    initial_parameters = np.hstack((X.ravel(), Theta.ravel()))
    lambda_coef = 10
    theta = minimize(cofi_J, initial_parameters, method='L-BFGS-B', jac=cofi_grad, options={'disp': True, 'maxiter': 150},
                        args=(Y, R, num_users, num_movies, num_features, lambda_coef)).x

    X = theta[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

    p = X.dot(Theta.T)
    my_predictions = p[:, :1] + Ymean

    movie_list = load_movie_list()

    ix = np.argsort(my_predictions.ravel())[::-1]

    print('Top recommendations for you:')
    for i in range(10):
        j = ix[i]
        print('Predicting rating %s for movie %s' % (my_predictions[j], movie_list[j]))

    print('\nOriginal ratings provided:')
    for my_rating, movie in zip(my_ratings.ravel(), movie_list):
        if my_rating > 0:
            print('Rated %s for %s' % (my_rating, movie))
