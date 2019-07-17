import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

file_path = 'data.mat'

def load_data(path):
    data = scio.loadmat(path)['data']
    return data

def kmeans_clustering(data, num):
    tic = time.time()
    init_idx = np.random.choice(data.shape[0], num)
    centers = data[init_idx, :]
    distances = np.empty((data.shape[0], num))
    last_labels = np.zeros(data.shape[0])
    labels = np.ones(data.shape[0])
    while np.any(last_labels != labels):
        last_labels = labels
        for i in range(num):
            distances[:, i] = np.sum(np.square(data - centers[i]), axis=1)
        labels = np.argmin(distances, axis=1)
        for i in range(num):
            if np.any(labels == i):
                centers[i] = np.mean(data[labels == i], axis=0)
    t = time.time() - tic
    return init_idx, labels, t

def get_silhouette_coefficient(data, labels, c_num):
    labeled_data = [None] * c_num
    a_score = np.zeros((c_num, ))
    b_score = np.zeros((c_num, ))
    for i in range(c_num):
        labeled_data[i] = data[labels == i]
    for i in range(c_num):
        sum_a = 0.0
        for j in range(labeled_data[i].shape[0]):
            sum_a += np.sum(np.sqrt(np.sum(np.square(labeled_data[i] - labeled_data[i][j, :]), axis=1)), axis=0) \
                     / (labeled_data[i].shape[0] - 1)
        a_score[i] = sum_a / labeled_data[i].shape[0]
        b_min = np.inf
        for j in range(c_num):
            if i != j:
                sum_b = 0.0
                for k in range(labeled_data[i].shape[0]):
                    sum_b += np.mean(np.sqrt(np.sum(np.square(labeled_data[j] - labeled_data[i][k, :]), axis=1)), axis=0)
                b_min = sum_b / labeled_data[i].shape[0] if b_min > sum_b / labeled_data[i].shape[0] else b_min
        b_score[i] = b_min
    return np.mean(np.divide(b_score - a_score, np.maximum(a_score, b_score)))

if __name__ == '__main__':
    classes = np.arange(2, 8)
    data = load_data(file_path)
    ''' test with different num of class '''
    silhouette_coefficient = [None] * classes.shape[0]
    plt.figure(figsize=(12, 8))
    plt.suptitle('K-means Cluster', fontsize=14)
    gs = gridspec.GridSpec(2, 3)
    for c_num in classes:
        ax = plt.subplot(gs[int((c_num - 2) / 3), int(c_num - 2) % int(3)])
        ax.set_title('class num = %d' % c_num)
        init_idx, labels, _ = kmeans_clustering(data, c_num)
        for i in range(c_num):
            labeled_data = data[labels == i]
            ax.scatter(labeled_data[:, 0], labeled_data[:, 1], s=2)
            ax.scatter(data[init_idx, 0], data[init_idx, 1], marker='x', s=12, c='black')
        silhouette_coefficient[c_num - 2] = get_silhouette_coefficient(data, labels, c_num)
    plt.show()
    plt.figure()
    plt.title('silhouette coefficient of different num of class')
    plt.xlabel('num of class')
    plt.ylabel('silhouette coefficient')
    plt.plot(classes, silhouette_coefficient)
    plt.grid()
    plt.show()

    ''' test with different initial points '''
    c_num = 7
    plt.figure(figsize=(11, 5))
    plt.suptitle('test with different initial points(num of class = %d)' % c_num)
    ax = plt.subplot('121')
    init_idx, labels, _ = kmeans_clustering(data, c_num)
    for i in range(c_num):
        labeled_data = data[labels == i]
        ax.scatter(labeled_data[:, 0], labeled_data[:, 1], s=2)
        ax.scatter(data[init_idx, 0], data[init_idx, 1], marker='x', s=12, c='black')
    ax = plt.subplot('122')
    init_idx, labels, _ = kmeans_clustering(data, c_num)
    for i in range(c_num):
        labeled_data = data[labels == i]
        ax.scatter(labeled_data[:, 0], labeled_data[:, 1], s=2)
        ax.scatter(data[init_idx, 0], data[init_idx, 1], marker='x', s=12, c='black')
    plt.show()

    ''' test with different amount of data '''
    data_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    elapsed = np.zeros(len(data_ratios))
    epoches = 100
    for epoch in range(epoches):
        for i in range(len(data_ratios)):
            part = data[:int(data_ratios[i] * data.shape[0]), :]
            _, _, t = kmeans_clustering(part, 3)
            elapsed[i] += t
    elapsed = elapsed / epoches
    plt.figure()
    plt.title('time elapsed of different ratio of data')
    plt.xlabel('data ratio')
    plt.ylabel('time(ms)')
    plt.plot(data_ratios, elapsed)
    plt.grid()
    plt.show()
