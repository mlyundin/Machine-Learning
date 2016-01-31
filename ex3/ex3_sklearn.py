import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data = sio.loadmat('ex3data1.mat')

    y = data['y']
    X = data['X']

    # replace 10 by 0
    y = y % 10

    n_sampels = 100
    sampels = np.random.choice(len(X), n_sampels)

    fig = plt.figure(figsize=(8, 8))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i, j in enumerate(sampels):
        ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[j, :].reshape(20, 20).T, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(y[j, 0]))

    plt.show()

    y = y.ravel()
    lr = LogisticRegression(C=100)
    lr.fit(X, y)

    print 'Training Set Accuracy: {}'.format(lr.score(X, y)*100)

    # Use regularization
    lr = LogisticRegression(C=10)
    lr.fit(X, y)

    print 'Training Set Accuracy: {}'.format(lr.score(X, y)*100)