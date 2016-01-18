import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv

from common_functions import load_data, J_liner_regression, add_zero_feature, gradient_descent, matrix_args, feature_normalize

if __name__ == '__main__':
    X, y = load_data('ex1data2.txt')

    mu, sigma, X = feature_normalize(X)
    X = add_zero_feature(X)

    iterations = 400
    alphas = [0.01, 0.1]
    f, axarr = plt.subplots(len(alphas), sharex=True)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J')
    for i, alpha in enumerate(alphas):
        theta = np.zeros((X.shape[1], 1))
        theta, J_history = gradient_descent(J_liner_regression, X, y, iterations, theta, alpha)

        axarr[i].set_title('Alpha = {}'.format(alpha))
        axarr[i].plot(range(len(J_history)), J_history)

    plt.show()
    # % Estimate the price of a 1650 sq-ft, 3 br house
    # % ====================== YOUR CODE HERE ======================
    # % Recall that the first column of X is all-ones. Thus, it does
    # % not need to be normalized.

    x = np.ones((1, X.shape[1]))
    x[:, 1:] = (np.array([[1650, 3]])-mu)/sigma
    prize = np.dot(x, theta)[0, 0]

    print prize

    X, y = load_data('ex1data2.txt')
    X = add_zero_feature(X)

    @matrix_args
    def calc_theta(X, y):
        return pinv(X.T*X)*X.T*y
    theta = calc_theta(X, y)

    prize = np.dot(np.array([[1, 1650, 3]]), theta)[0, 0]

    print prize