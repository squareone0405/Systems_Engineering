import numpy as np
import math
from scipy import stats

def load_data():
    X = np.array([[149.3, 4.2, 80.3, 108.1],
                  [161.2, 4.1, 72.9, 114.8],
                  [171.5, 3.1, 45.6, 123.2],
                  [175.5, 3.1, 50.2, 126.9],
                  [180.8, 1.1, 68.8, 132.0],
                  [190.7, 2.2, 88.5, 137.7],
                  [202.1, 2.1, 87.0, 146.0],
                  [212.4, 5.6, 96.9, 154.1],
                  [226.1, 5.0, 84.9, 162.3],
                  [231.9, 5.1, 60.7, 164.3],
                  [239.0, 0.7, 70.4, 167.6]])
    Y = np.array([15.9, 16.4, 19.0, 19.1, 18.88, 20.4,
                  22.7, 26.5, 28.1, 27.6, 26.3])
    return X, Y

def linear_regression(Y, X, alpha, feature_preserve):
    X_bar = np.average(X, axis=0)
    X_sigma = np.sqrt(np.sum(np.multiply(X - X_bar, X - X_bar), axis=0) / X.shape[0])
    X_norm = (X - X_bar) / X_sigma
    y_bar = np.average(Y)
    y_sigma = np.sqrt(np.sum(np.multiply(Y - y_bar, Y - y_bar)) / Y.size)
    Y_norm = (Y - y_bar) / y_sigma
    eigen_values, eigen_vectors = np.linalg.eigh(np.matmul(X_norm.transpose(), X_norm))
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    ev_sum = np.sum(eigen_values)
    temp = 0.0
    counter = 0
    while temp / ev_sum < feature_preserve:
        temp += eigen_values[counter]
        counter += 1
    if counter < X.shape[1]:
        print('dimension compressed from %d to %d' % (X.shape[1], counter))
    Qm = eigen_vectors[:, :counter].transpose()
    c = np.matmul(np.matmul(np.matmul(Qm.transpose(), (1.0 / eigen_values[:counter]).reshape((counter, 1)) * Qm),
                            X_norm.transpose()), Y_norm)
    weight = c * y_sigma / X_sigma
    bias = y_bar - np.matmul(weight,  X_bar)
    result_str = 'y = '
    for i in range(X.shape[1]):
        result_str += '(%.4f*x%d) + ' % (weight[i], i + 1)
    result_str += '(%.4f)' % (bias)
    print(result_str)
    Y_hat = np.matmul(weight.reshape((1, X.shape[1])), X.transpose()) + bias
    ESS = np.sum(np.multiply(Y_hat - y_bar, Y_hat - y_bar))
    RSS = np.sum(np.multiply(Y - Y_hat, Y - Y_hat))
    F = (X.shape[0] - X.shape[1] - 1) * ESS / (X.shape[1] * RSS)
    F_alpha = stats.f.isf(q=alpha, dfn=X.shape[1], dfd=X.shape[0] - X.shape[1] - 1)
    print('F = %.2f' % (F))
    print('F_alpha = %.2f' % (F_alpha))
    if F > F_alpha:
        print('Y and X have linear relationship')
        S_sigma = math.sqrt(RSS / (X.shape[0] - X.shape[1] - 1))
        interval = S_sigma * stats.norm.isf(alpha / 2)
        print('confidence interval is %.4f' % (interval))
    else:
        print('Y and X do not have linear relationship')

if __name__ == '__main__':
    X, Y = load_data()
    linear_regression(Y, X, 0.05, 0.90)
