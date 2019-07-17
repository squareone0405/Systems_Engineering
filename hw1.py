# coding:utf-8

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] # to show chinese label
data_path = './data.mat'

def movingAverage(data, N):
    data_len = data.shape[0]
    data_average = [None] * data_len
    sum = 0.0
    for i in range(data_len):
        sum += data[i]
        if i >= N:
            sum -= data[i - N]
            data_average[i] = sum / N
        else:
            data_average[i] = sum / (i + 1)
    return data_average

def exponentialSmoothing(data, alpha):
    data_len = data.shape[0]
    data_exp = [None] * data_len
    data_exp[0] = data[0]
    for i in np.arange(1, data_len):
        data_exp[i] = data_exp[i - 1] * (1 - alpha) + data[i] * alpha
    return data_exp

if __name__ == '__main__':
    data_file = scio.loadmat(data_path)
    data = data_file['data']
    len = data.shape[0]
    x = np.arange(0, len) * 30 / 3600

    ''' origin data '''
    plt.figure()
    plt.title('高速公路车流量图', fontsize=18)
    plt.plot(x, data, color='blue', linewidth=1, linestyle='-')
    plt.xlabel('时间(小时)', fontsize=14)
    plt.ylabel('车流量(辆/小时)', fontsize=14)
    plt.xlim(0, len * 30 / 3600)
    plt.ylim(0)
    plt.grid()

    ''' moving average '''
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('高速公路车流量图', fontsize=18)
    ax1 = plt.subplot(121)
    ax1.set_title(r'$N=10$', fontsize=15)
    ax1.plot(x, movingAverage(data, 10), color='green', linewidth=1, linestyle='-')
    ax1.set_xlabel('时间(小时)', fontsize=14)
    ax1.set_ylabel('车流量(辆/小时)', fontsize=14)
    ax1.set_xlim(0, len * 30 / 3600)
    ax1.set_ylim(0)
    ax1.grid()

    ax2 = plt.subplot(122)
    ax2.set_title(r'$N=30$', fontsize=15)
    ax2.plot(x, movingAverage(data, 30), color='red', linewidth=1, linestyle='-')
    ax2.set_xlabel('时间(小时)', fontsize=14)
    ax2.set_ylabel('车流量(辆/小时)', fontsize=14)
    ax2.set_xlim(0, len * 30 / 3600)
    ax2.set_ylim(0)
    ax2.grid()

    ''' exponential smoothing '''
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('高速公路车流量图', fontsize=18)
    ax1 = plt.subplot(121)
    ax1.set_title(r'$\alpha=0.2$', fontsize=15)
    ax1.plot(x, exponentialSmoothing(data, 0.2), color='green', linewidth=1, linestyle='-')
    ax1.set_xlabel('时间(小时)', fontsize=14)
    ax1.set_ylabel('车流量(辆/小时)', fontsize=14)
    ax1.set_xlim(0, len * 30 / 3600)
    ax1.set_ylim(0)
    ax1.grid()

    ax2 = plt.subplot(122)
    ax2.set_title(r'$\alpha=0.05$', fontsize=15)
    ax2.plot(x, exponentialSmoothing(data, 0.05), color='red', linewidth=1, linestyle='-')
    ax2.set_xlabel('时间(小时)', fontsize=14)
    ax2.set_ylabel('车流量(辆/小时)', fontsize=14)
    ax2.set_xlim(0, len * 30 / 3600)
    ax2.set_ylim(0)
    ax2.grid()
    plt.show()
