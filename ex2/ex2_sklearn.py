import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from common_functions import load_data

if __name__ == '__main__':
    X, y = load_data('ex2data1.txt')

    x1, x2 = X.T
    f_y = y.ravel()
    plt.plot(x1[f_y==0], x2[f_y==0], 'yo')
    plt.plot(x1[f_y==1], x2[f_y==1], 'bx')

    plt.show()

    lr = LogisticRegression(C=100)
    lr.fit(X, f_y)
    theta = np.array([lr.intercept_[0], lr.coef_[0, 0], lr.coef_[0, 1]])

    x1_boundery = np.array([np.min(x1)-2, np.max(x1)+2])
    x2_boundery = (-1/theta[2])*(theta[1]*x1_boundery + theta[0])

    plt.plot(x1[f_y==0], x2[f_y==0], 'yo')
    plt.plot(x1[f_y==1], x2[f_y==1], 'bx')
    plt.plot(x1_boundery, x2_boundery)
    plt.show()


    print 'Train Accuracy: {}%'.format(lr.score(X, y)*100)
