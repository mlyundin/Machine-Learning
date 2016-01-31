import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import numpy as np
import scipy.io as sio

from ex7_pca import project_data, recover_data, display_data

if __name__ == '__main__':

    data = sio.loadmat('ex7data1.mat')
    X = data['X']
    x1, x2 = X.T

    plt.plot(x1, x2, 'bo')
    plt.show()

    ss = StandardScaler()
    pca = PCA()
    pipeline = Pipeline([("standart_scaler", ss),
                         ("PCA", pca)])
    pipeline.fit(X)

    print('Top eigenvector: ')
    print(' components_[0, :] = %s' % pca.components_[0, :])
    print('(you should expect to see 0.707107 0.707107)')

    X_norm = ss.transform(X)

    K = 1
    U = pca.components_.T
    Z = project_data(X_norm, U, K)
    print('Projection of the first example: %s' % Z[0])
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

    data = sio.loadmat('ex7faces.mat')
    X = data['X']

    display_data(X[:100, :], 'Faces dataset')

    pipeline.fit(X)
    X_norm = ss.transform(X)

    U = pca.components_.T

    display_data(U[:, :36].T, 'Principal components on the face dataset')

    K = 100
    Z = project_data(X_norm, U, K)
    X_rec = recover_data(Z, U, K)

    display_data(X_norm[:100, :], 'Original images of faces')
    display_data(X_rec[:100, :], 'Reconstructed from only the top 100 principal components')

    #PCA for Visualization
    img=mpimg.imread('bird_small.png')
    img_size = img.shape

    X = img.reshape(img_size[0]*img_size[1], img_size[2])

    K = 16
    max_iters = 10

    km = KMeans(K, max_iter=max_iters)
    km.fit(X)
    to_plot = range(len(X))
    np.random.shuffle(to_plot)
    to_plot = to_plot[:1000]

    X_plot = X[to_plot, :]
    idx_plot = km.labels_[to_plot]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k, color in zip(xrange(K), itertools.cycle(['r', 'b', 'g', 'k', 'm'])):
        x, y, z = X_plot[idx_plot == k, :].T
        ax.scatter(x, y, z, c=color)
    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.show()

    pipeline.fit(X)
    X_norm = ss.transform(X)
    U = pca.components_.T
    Z = project_data(X_norm, U, 2)
    Z_plot = Z[to_plot, :]
    for k, color in zip(xrange(K), itertools.cycle(['r', 'b', 'g', 'k', 'm'])):
        x1, x2 = Z_plot[idx_plot == k, :].T
        plt.plot(x1, x2, color+'o')
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()

