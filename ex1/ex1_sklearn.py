import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from common_functions import load_data

if __name__ == '__main__':
    X, y = load_data('ex1data1.txt')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    plt.plot(X, y, 'rx')
    plt.show()

    reg = LinearRegression()
    reg.fit(X, y)

    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.plot(X, reg.predict(X))
    plt.plot(X, y, 'rx')
    plt.show()