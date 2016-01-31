import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import scipy.io as sio
import itertools

def find_closest_centroids(X, centroids):
    return np.array([np.argmin([np.sum((x-c)**2) for c in centroids]) for x in X])

def compute_centroids(X, idx, K):
    return np.array([np.mean(X[idx.ravel() == i, :], axis=0) for i in range(K)])

def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    previous_centroids = initial_centroids
    K = len(previous_centroids)
    for i in xrange(max_iters):
        idx = find_closest_centroids(X, previous_centroids)
        centroids = compute_centroids(X, idx, K)
        if (centroids == previous_centroids).all():
            break

        if plot_progress:
            for k in xrange(K):
                x1 = [previous_centroids[k, 0], centroids[k, 0]]
                x2 = [previous_centroids[k, 1], centroids[k, 1]]

                plt.plot(x1, x2, 'k')
                plt.plot(x1, x2, 'kx')

        previous_centroids = centroids

    if plot_progress:
        for k, color in zip(xrange(K), itertools.cycle(['r', 'b', 'g', 'k', 'm'])):
            x1, x2 = X.T
            plt.plot(x1[idx == k], x2[idx == k], color+'o')
        plt.show()

    return idx, centroids


def kmeans_init_centroids(X, K):
    temp = range(len(X))
    np.random.shuffle(temp)

    return X[temp[:K], :]

if __name__ == '__main__':

    data = sio.loadmat('ex7data2.mat')
    X = data['X']

    K = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    idx = find_closest_centroids(X, initial_centroids)

    print('Closest centroids for the first 3 examples: ')
    print(idx[:3])
    print('(the closest centroids should be 0, 2, 1 respectively)')

    idx, centroids = run_kmeans(X, initial_centroids, 400, True)

    img=mpimg.imread('bird_small.png')
    imgplot = plt.imshow(img)
    plt.show()

    img_size = img.shape
    X = img.reshape(img_size[0]*img_size[1], img_size[2])
    K = 16
    max_iters = 10

    initial_centroids = kmeans_init_centroids(X, K)
    idx, centroids = run_kmeans(X, initial_centroids, max_iters)

    X_recovered = centroids[idx, :]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img)
    ax2.imshow(X_recovered.reshape(img_size))

    plt.show()