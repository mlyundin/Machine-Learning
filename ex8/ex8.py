import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import det, pinv
import scipy.io as sio

def visualize_data(X, title):
    x1, x2 = X.T
    plt.plot(x1, x2, 'bx')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.xlim([0, 30])
    plt.ylim([0, 30])
    plt.title(title)
    return plt

def visualize_fit(X, mu, sigma2):
    visualize_data(X, 'Visualizing Gaussian fit.')
    x = np.arange(0, 35, 0.5)
    x1, x2 = np.meshgrid(x, x)

    z = multivariate_gaussian(np.hstack((x1.reshape(-1,1), x2.reshape(-1,1))), mu, sigma2).reshape(x1.shape)
    plt.contour(x1, x2, z)

    return plt

def estimate_gaussian(X):

    return np.mean(X, axis=0)[:, np.newaxis], np.var(X, axis=0)[:, np.newaxis]

def multivariate_gaussian(X, mu, Sigma2):
    k = float(len(mu))
    X = np.copy(X)

    if any(s == 1 for s in Sigma2.shape):
        Sigma2 = np.diag(Sigma2.ravel())

    X -= mu.reshape(1, -1)
    return (2*np.pi)**(-k/2)*det(Sigma2)**(-0.5)*np.exp(-0.5*np.sum(np.dot(X, pinv(Sigma2))*X, axis=1))

def select_threshold(yval, pval):
    yval = yval.ravel()

    best_epsilon = 0
    best_F1 = 0
    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        cvPredictions = pval < epsilon

        tp = np.sum((cvPredictions == 1) & (yval == 1), dtype=float)
        fp = np.sum((cvPredictions == 1) & (yval == 0))
        fn = np.sum((cvPredictions == 0) & (yval == 1))
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)

        F1 = 2*recall*precision/(recall+precision)

        if F1 > best_F1:
            best_F1, best_epsilon = F1, epsilon

    return best_epsilon, best_F1

if __name__ == '__main__':
    data = sio.loadmat('ex8data1.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    visualize_data(X, 'Visualizing example dataset for outlier detection').show()

    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    visualize_fit(X, mu, sigma2).show()

    pval = multivariate_gaussian(Xval, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)

    print('Best epsilon found using cross-validation: %s' % epsilon)
    print('Best F1 on Cross Validation Set:  %s' % F1)
    print('   (you should see a value epsilon of about 8.99e-05)')

    visualize_data(X, 'The classified anomalies.')
    x1, x2 = X[p < epsilon, :].T
    plt.plot(x1, x2, 'ro')
    plt.show()

    data = sio.loadmat('ex8data2.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    mu, sigma2 = estimate_gaussian(X)

    p = multivariate_gaussian(X, mu, sigma2)
    pval = multivariate_gaussian(Xval, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)

    print('Best epsilon found using cross-validation: %s' % epsilon)
    print('Best F1 on Cross Validation Set:  %s' % F1)
    print('# Outliers found: %s' % np.sum(p < epsilon))
    print('   (you should see a value epsilon of about 1.38e-18)')