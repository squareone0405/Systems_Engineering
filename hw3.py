import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, f

def load_data():
    X = np.array([0.009, 0.013, 0.006, 0.025, 0.022, 0.007, 0.036,
                  0.014, 0.016, 0.014, 0.016, 0.012, 0.020, 0.018])
    Y = np.array([4.0, 3.44, 3.6, 1.0, 2.04, 4.74, 0.6,
                  1.7, 2.92, 4.8, 3.28, 4.16, 3.35, 2.2])
    return X, Y

def linear_regression(data, alpha):
    Y = data[0]
    X = data[1]
    xbar = np.average(X)
    ybar = np.average(Y)
    Lxy = np.sum(np.multiply(X - xbar, Y - ybar))
    Lxx = np.sum(np.multiply(X - xbar, X - xbar))
    Lyy = np.sum(np.multiply(Y - ybar, Y - ybar))
    b = Lxy / Lxx
    a = ybar - b * xbar
    print(r'y = %.2fx + %.2f' % (b, a))
    Yhat = a + b * X
    TSS = Lyy
    ESS = np.sum(np.multiply(Yhat - ybar, Yhat - ybar))
    RSS = np.sum(np.multiply(Y - Yhat, Y - Yhat))
    r2 = ESS / TSS
    r = math.sqrt(r2)
    r = r if b > 0 else -r
    print(r'r = %.3f' % (r))
    F = (len(X) - 2) * ESS / RSS
    print(r'F = %.2f' % (F))
    f_thres = f.isf(q=alpha, dfn=1, dfd=(len(X) - 2))
    print(r'F_\alpha = %.2f' % (f_thres))
    if F > f_thres:
        S_sigma = math.sqrt(RSS / (len(X) - 2))
        interval = S_sigma * norm.isf(alpha / 2)
        plt.figure(figsize=(8, 6))
        plt.scatter(X, Y, c='red', marker='+', label='origin data')
        plt.plot(X, Yhat, label=r'$y = %.2f x + %.2f$' % (b, a))
        plt.plot(X, Yhat + interval,
                 label=r'$y = %.2f x + %.2f(interval = %.2f)$' % (b, a + interval, interval), linestyle=':')
        plt.plot(X, Yhat - interval,
                 label=r'$y = %.2f x + %.2f(interval = %.2f)$' % (b, a - interval, interval), linestyle=':')
        plt.title(r'Linear Regression($F = %.2f > %.2f = F_\alpha$)' % (F, f_thres))
        plt.xlabel(r'x')
        plt.ylabel(r'y')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print('The data do not satisfy the condition of linear relationship')

if __name__ == '__main__':
    X, Y = load_data()
    linear_regression([Y, X], 0.05)

