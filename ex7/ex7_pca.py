import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.linalg import svd
import scipy.io as sio
import itertools

from common_functions import feature_normalize, matrix_args, matrix_args_array_only
from ex7 import kmeans_init_centroids, run_kmeans

@matrix_args
def pca(X):
    m = len(X)

    u, s, v = svd(X.T*X/m)

    return u, s

@matrix_args_array_only
def project_data(X, U, K):
    return X*U[:, :K]

@matrix_args_array_only
def recover_data(Z, U, K):
    return (Z*(U[:, :K]).T).A

def display_data(X, title):
    fig = plt.figure(figsize=(8, 8))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    plt.suptitle(title, fontsize=18, color='r')

    for i, x in enumerate(X):
        ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(x.reshape(32, 32).T, cmap=plt.cm.Greys_r, interpolation='nearest')

    plt.show()


if __name__ == '__main__':

    data = sio.loadmat('ex7data1.mat')
    X = data['X']
    x1, x2 = X.T

    plt.plot(x1, x2, 'bo')
    plt.show()

    mu, sigma, X_norm = feature_normalize(X)

    U, S = pca(X_norm)

    print('Top eigenvector: ');
    print(' U[:,0] = %s' % U[:, 0])
    print('(you should expect to see -0.707107 -0.707107)')

    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example: %f', Z[0])
    print('(this value should be about 1.49631261)')

    X_rec = recover_data(Z, U, K)
    print('Approximation of the first example: %s' % X_rec[0])
    print('(this value should be about  -1.05805279 -1.05805279)')

    x1, x2 = X_norm.T
    plt.plot(x1, x2, 'bo')
    x1, x2 = X_rec.T
    plt.plot(x1, x2, 'ro')
    plt.axis('equal')

    for x_rec, x_norm in zip(X_rec, X_norm):
        plt.plot((x_rec[0], x_norm[0]), (x_rec[1], x_norm[1]), 'k-')
    plt.show()

    #PCA on Face Data: Eigenfaces
    data = sio.loadmat('ex7faces.mat')
    X = data['X']

    display_data(X[:100, :], 'Faces dataset')

    mu, sigma, X_norm = feature_normalize(X)

    U, S = pca(X_norm)

    display_data(U[:, :36].T, 'Principal components on the face dataset')

    K = 100
    Z = project_data(X_norm, U, K)
    X_rec = recover_data(Z, U, K)

    display_data(X_norm[:100, :], 'Original images of faces')
    display_data(X_rec[:100, :], 'Reconstructed from only the top 100 principal components')

    #PCA for Visualization
    img=mpimg.imread('bird_small.png')
    img_size = img.shape

    X = img.reshape(img_size[0]*img_size[1], 3)

    K = 16
    max_iters = 10
    initial_centroids = kmeans_init_centroids(X, K)
    idx, centroids = run_kmeans(X, initial_centroids, max_iters)

    to_plot = range(len(X))
    np.random.shuffle(to_plot)
    to_plot = to_plot[:1000]

    X_plot = X[to_plot, :]
    idx_plot = idx[to_plot]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k, color in zip(xrange(K), itertools.cycle(['r', 'b', 'g', 'k', 'm'])):

        x, y, z = X_plot[idx_plot == k, :].T
        ax.scatter(x, y, z, c=color)
    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.show()

    mu, sigma, X_norm = feature_normalize(X)
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2)

    Z_plot = Z[to_plot, :]
    for k, color in zip(xrange(K), itertools.cycle(['r', 'b', 'g', 'k', 'm'])):

        x1, x2 = Z_plot[idx_plot == k, :].T
        plt.plot(x1, x2, color+'o')
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()
