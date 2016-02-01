import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from collections import namedtuple
from functools import partial

Model = namedtuple('SVMModelResult', ['X', 'y', 'kernelFunction', 'b', 'alphas', 'w'])

def svm_train(X, Y, C, kernelFunction, tol=0.001, max_passes=5):

    m, n = X.shape
    Y = Y.copy()
    Y[Y == 0] = -1

    alphas = np.zeros((m, 1))
    b = 0.0
    E = np.zeros((m, 1))
    passes = 0.0
    eta = 0.0
    L = 0.0
    H = 0.0

    kernelFunctionName = kernelFunction.__name__

    if kernelFunctionName == 'linear_kernel':
        K = X.dot(X.T)
    elif kernelFunctionName == 'gaussian_kernel':
        X2 = np.sum(X**2, axis=1)[:, np.newaxis]
        K = X2 + (X2.T - 2*X.dot(X.T))
        K = kernelFunction(1,0)**K
    else:
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K[i,j] = K[j, i] = kernelFunction(X[i:i+1, :].T, X[j:j+1, :].T)

    while passes < max_passes:

        num_changed_alphas = 0
        for i in range(m):

            E[i] = b + np.sum(alphas*Y*K[:, [i]]) - Y[i, 0]

            if (Y[i, 0]*E[i, 0] < -tol and alphas[i, 0] < C) or (Y[i, 0]*E[i, 0] > tol and alphas[i, 0] > 0):

                while True:
                    j = np.random.randint(0, m)
                    if j != i:
                        break
                E[j] = b + np.sum(alphas*Y*K[:, j:j+1]) - Y[j, 0]

                alpha_i_old = alphas[i, 0]
                alpha_j_old = alphas[j, 0]

                if Y[i] == Y[j]:
                    L = max(0.0, alpha_j_old + alpha_i_old - C)
                    H = min(C, alpha_j_old + alpha_i_old)
                else:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(C, C + alpha_j_old - alpha_i_old)

                if L == H:
                    continue

                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue


                alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta
                alphas[j] = np.minimum(H, alphas[j,0])
                alphas[j] = np.maximum(L, alphas[j,0])

                if (np.abs(alphas[j,0] - alpha_j_old) < tol):
                    alphas[j] = alpha_j_old
                    continue

                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])

                b1 = (b - E[i] - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j]- Y[j] * (alphas[j] - alpha_j_old) *  K[i,j])[0]
                b2 = (b - E[j] - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j]- Y[j] * (alphas[j] - alpha_j_old) *  K[j,j])[0]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j]< C:
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    idx = alphas.ravel() > 0
    return Model(X[idx,:], Y[idx, :], kernelFunction, b, alphas[idx, :], np.dot((alphas*Y).T, X).T)

def svm_predict(model, X):
    if X.shape[1] == 1:
        X = X.T

    m = len(X)
    pred = np.zeros(m)

    kernelFunctionName = model.kernelFunction.__name__

    if kernelFunctionName == 'linear_kernel':
        p = np.dot(X, model.w) + model.b
    elif kernelFunctionName == 'gaussian_kernel':
        X1 = np.sum(X**2, axis=1)[:, np.newaxis]
        X2 = np.sum(model.X**2, axis=1)[np.newaxis, :]
        K = X1 + (X2-2*np.dot(X, model.X.T)) #bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
        K = model.kernelFunction(1, 0) ** K
        K = model.y.T*K #bsxfun(@times, model.y', K);
        K = model.alphas.T*K #bsxfun(@times, model.alphas', K);
        p = np.sum(K, axis=1)

    pred[p.ravel() >= 0] = 1

    return pred

def visualize_boundary(X, y, model):
    visualize_data(X, y)
    x1, x2 = X.T
    x1plot = np.linspace(np.min(x1), np.max(x1), 100)
    x2plot = np.linspace(np.min(x2), np.max(x2), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros_like(X1)
    for i in range(X1.shape[1]):
        this_X = np.hstack((X1[:, i:i+1], X2[:, i:i+1]))
        vals[:, i] = svm_predict(model, this_X)

    CS = plt.contour(x1plot, x2plot, vals)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()

def visualize_data(X, y):
    x1, x2 = X.T
    f_y = y.ravel()
    plt.plot(x1[f_y==0], x2[f_y==0], 'yo')
    plt.plot(x1[f_y==1], x2[f_y==1], 'bx')

    return plt

def visualize_boundary_linear(X, y, model):
    visualize_data(X, y)
    if model:
        x1 = X[:, 0]
        w = model.w
        b = model.b
        xp = np.linspace(np.min(x1), np.max(x1), 100)
        yp = - (w[0]*xp + b)/w[1]
        plt.plot(xp, yp)

    plt.show()

def linear_kernel(x1, x2):

    return x1.dot(x2)

def gaussian_kernel(x1, x2, sigma=1.0):

    return np.exp(-np.sum((x1-x2)**2)/(2*float(sigma)**2))

if __name__ == '__main__':
    data = sio.loadmat('ex6data1.mat')
    y = data['y'].astype(np.float64)
    X = data['X']

    visualize_boundary_linear(X, y, None)

    C = 1
    model = svm_train(X, y, C, linear_kernel, 0.001, 20)
    visualize_boundary_linear(X, y, model)

    C = 100
    model = svm_train(X, y, C, linear_kernel, 0.001, 20)
    visualize_boundary_linear(X, y, model)

    x1 = np.array([1, 2, 1], dtype=np.float64)
    x2 = np.array([0, 4, -1], dtype=np.float64)
    sigma = 2.0
    sim = gaussian_kernel(x1, x2, sigma);

    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 : (this value should be about 0.324652)')
    print('Actual = {}'.format(sim))

    data = sio.loadmat('ex6data2.mat')
    y = data['y'].astype(np.float64)
    X = data['X']

    visualize_data(X, y).show()

    C = 1.0
    sigma = 0.1

    partialGaussianKernel = partial(gaussian_kernel, sigma=sigma)
    partialGaussianKernel.__name__ = gaussian_kernel.__name__
    model= svm_train(X, y, C, partialGaussianKernel)
    visualize_boundary(X, y, model)

    data = sio.loadmat('ex6data3.mat')
    y = data['y'].astype(np.float64)
    X = data['X']
    Xval = data['Xval']
    yval = data['yval'].astype(np.float64)

    visualize_data(X, y).show()

    best_C = 0
    best_sigma = 0
    best_error = len(yval)
    best_model = None

    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            partialGaussianKernel = partial(gaussian_kernel, sigma=sigma)
            partialGaussianKernel.__name__ = gaussian_kernel.__name__

            model= svm_train(X, y, C, partialGaussianKernel)
            ypred = svm_predict(model, Xval)
            error = np.mean(ypred != yval.ravel())
            if error < best_error:
                best_error = error
                best_C = C
                best_sigma = sigma
                best_model = model

    visualize_boundary(X, y, best_model)