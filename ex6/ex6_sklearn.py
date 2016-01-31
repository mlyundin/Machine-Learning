import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV

def visualize_boundary(X, y, clfs={}):

    if clfs:
        h = 0.02
        scale = 0.1
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        x1_min, x1_max = x1_min - scale*np.abs(x1_min), x1_max + scale*np.abs(x1_max)
        x2_min, x2_max = x2_min - scale*np.abs(x2_min), x2_max + scale*np.abs(x2_max)

        X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                        np.arange(x2_min, x2_max, h))

        n_of_row, n_of_col = (1, 1) if len(clfs) == 1 else ((len(clfs)+1) / 2, 2)
        for i, clf in enumerate(clfs.iteritems()):
            title, clf = clf
            plt.subplot(n_of_row, n_of_col, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
            plt.contourf(X1, X2, Z, cmap=plt.cm.Paired, alpha=0.8)

            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            plt.axis('equal')
            plt.title(title)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    plt.show()

def sigma_to_gamma(sigma):
    return 1.0/(sigma**2)

if __name__ == '__main__':
    data = sio.loadmat('ex6data1.mat')
    y = data['y'].astype(np.float64).ravel()
    X = data['X']
    visualize_boundary(X, y, None)

    C = 1
    lsvc = LinearSVC(C=C, tol=0.001)
    lsvc.fit(X, y)
    svc = SVC(C=C, tol=0.001, kernel='linear')
    svc.fit(X, y)
    visualize_boundary(X, y, {'SVM(linear kernel) C = {}'.format(C): svc,
                              'LinearSVC C = {}'.format(C): lsvc})

    C = 100
    lsvc = LinearSVC(C=C, tol=0.001)
    lsvc.fit(X, y)
    svc = SVC(C=C, tol=0.001, kernel='linear')
    svc.fit(X, y)
    visualize_boundary(X, y, {'SVM(linear kernel) C = {}'.format(C): svc,
                              'LinearSVC C = {}'.format(C): lsvc})

    data = sio.loadmat('ex6data2.mat')
    y = data['y'].astype(np.float64).ravel()
    X = data['X']

    visualize_boundary(X, y)

    C = 1.0
    sigma = 0.1
    gamma = sigma_to_gamma(sigma)
    svc = SVC(C=C, tol=0.001, kernel='rbf', gamma=gamma)
    svc.fit(X, y)
    visualize_boundary(X, y, {'SVM(rbf kernel) C = {}'.format(C): svc})

    data = sio.loadmat('ex6data3.mat')
    y = data['y'].astype(np.float64).ravel()
    X = data['X']
    Xval = data['Xval']
    yval = data['yval'].astype(np.float64).ravel()

    visualize_boundary(X, y)

    best_score = 0
    best_model = None

    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:

            gamma = sigma_to_gamma(sigma)
            svc = SVC(C=C, tol=0.001, kernel='rbf', gamma=gamma)
            svc.fit(X, y)

            score = svc.score(Xval, yval)

            if score > best_score:
                best_model, best_score = svc, score


    visualize_boundary(X, y, {'Best model(C={}, gamma={})'.format(best_model.C, best_model.gamma): best_model})

    #Let's do the similar thing but using sklearn feature
    X_all = np.vstack((X, Xval))
    y_all = np.concatenate((y, yval))

    parameters = {'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], 'gamma': map(lambda x: 1.0/(x**2), [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])}
    svr = SVC(tol=0.001, kernel='rbf')
    clf = GridSearchCV(svr, parameters, cv=2)
    clf.fit(X_all, y_all)

    visualize_boundary(X, y, {'Best model(C={}, gamma={})'.format(clf.best_params_['C'], clf.best_params_['gamma']): clf})
