import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from common_functions import load_data, J_liner_regression, add_zero_feature, gradient_descent

if __name__ == '__main__':
    X, y = load_data('ex1data1.txt')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    plt.plot(X, y, 'rx')
    plt.show()

    X = add_zero_feature(X)

    theta = np.zeros((X.shape[1], 1))
    iterations = 1500
    alpha = 0.01
    theta, J_history = gradient_descent(J_liner_regression, X, y, iterations, theta, alpha)


    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.plot(X[:, 1], np.dot(X, theta))
    plt.plot(X[:, 1], y, 'rx')
    plt.show()

    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J')
    plt.plot(range(len(J_history)), J_history)
    plt.show()

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    p_x, p_y = np.meshgrid(theta0_vals, theta1_vals)

    p_z = np.zeros_like(p_x, dtype=np.float64)

    for i in range(p_x.shape[0]):
        for j in range(p_x.shape[1]):
            p_z[i, j] = J_liner_regression(X, y, np.array([p_x[i, j], p_y[i, j]])[:, np.newaxis])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(p_x, p_y, p_z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()