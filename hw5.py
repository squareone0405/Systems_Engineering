import numpy as np
import math
import scipy.io as scio
import scipy.stats as stats

data_path = './counties.mat'

def load_data(path):
    data_file = scio.loadmat(path)
    data = data_file['data']
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def pca_compress(data, rerr):
    X_bar = np.mean(data, axis=0)
    X_sigma = np.std(data, axis=0)
    X_norm = (data - X_bar) / X_sigma
    eigen_values, eigen_vectors = np.linalg.eigh(np.matmul(X_norm.transpose(), X_norm))
    ev_sum = np.sum(eigen_values)
    sum = 0.0
    counter = 0
    while sum / ev_sum < rerr:
        sum += eigen_values[counter]
        counter += 1
    counter -= 1
    if counter < X.shape[1]:
        print('dimension compressed from %d to %d' % (X.shape[1], X.shape[1] - counter))
    pcs = eigen_vectors[:, counter:]
    cprs_data = np.matmul(X_norm, pcs)
    cprs_c = [X_bar, X_sigma]
    return pcs, cprs_data, cprs_c

def pca_reconstruct(pcs, cprs_data, cprs_c):
    recon_data = cprs_data @ pcs.transpose() * cprs_c[1] + cprs_c[0]
    return recon_data

def linear_regresstion(X, y, rerr, alpha):
    pcs, cprs_data, cprs_c = pca_compress(X, rerr)
    recon_data = pca_reconstruct(pcs, cprs_data, cprs_c)
    y_bar = np.mean(y, axis=0)
    y_sigma = np.std(y, axis=0)
    y_norm = (y - y_bar) / y_sigma
    b_hat = np.linalg.inv(cprs_data.transpose() @ cprs_data) \
            @ cprs_data.transpose() @ y_norm.reshape(y_norm.size, 1)
    weight = pcs @ b_hat * y_sigma / cprs_c[1].reshape(-1, 1)
    bias = y_bar - cprs_c[0] @ weight
    result_str = 'y = '
    for i in range(pcs.shape[0]):
        result_str += '(%.6f * x%d) + ' % (weight[i], i + 1)
    result_str += '(%.6f)' % (bias)
    print(result_str)
    y_hat = (X @ weight + bias).squeeze()
    ESS = np.sum(np.multiply(y_hat - y_bar, y_hat - y_bar))
    RSS = np.sum(np.multiply(y - y_hat, y - y_hat))
    F = (X.shape[0] - X.shape[1] - 1) * ESS / (X.shape[1] * RSS)
    F_alpha = stats.f.isf(q=alpha, dfn=X.shape[1], dfd=X.shape[0] - X.shape[1] - 1)
    print('F = %.2f' % (F))
    print('F_alpha = %.2f' % (F_alpha))
    if F > F_alpha:
        print('Y and X have linear relationship')
        S_sigma = math.sqrt(RSS / (X.shape[0] - X.shape[1] - 1))
        interval = S_sigma * stats.norm.isf(alpha / 2)
        print('confidence interval is %.4f' % interval)
    else:
        print('Y and X do not have linear relationship')

if __name__ == '__main__':
    rerr = 0.05
    alpha = 0.05
    X, y = load_data(data_path)
    print(X.shape )
    linear_regresstion(X, y, rerr, alpha)
