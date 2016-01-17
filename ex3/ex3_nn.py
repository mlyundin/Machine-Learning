import scipy.io as sio
import numpy as np

from common_functions import add_zero_feature, sigmoid

if __name__ == '__main__':

    data = sio.loadmat('ex3data1.mat')
    y = data['y']
    X = data['X']

    X = add_zero_feature(X)

    data = sio.loadmat('ex3weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']

    p = sigmoid(np.dot(Theta1, X.T))
    p = add_zero_feature(p, axis=0)
    p = sigmoid(np.dot(Theta2, p))

    y_pred = np.argmax(p, axis=0)+1

    print 'Training Set Accuracy: {}'.format(np.mean(y_pred == y.flatten()) * 100)
