import matplotlib.pyplot as plt
import numpy as np


def load_data_set():
    data_mat, label_mat = [], []
    fr = open('../corpus/testSet-LR.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

def grad_descent(data_mat_in, class_labels):
    data_mat = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()

    m, n = np.shape(data_mat)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))  # y = x1*w1 + x2*w2 + x3*w3

    for k in range(max_cycles):
        h = sigmoid(data_mat*weights)
        grad = data_mat.transpose() * (h-label_mat)
        weights = weights - alpha * grad
    return weights


def plot_best_fit(weights):
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]

    x1_red, x2_red = [], []
    x1_blue, x2_blue = [], []

    for i in range(n):
        if int(label_mat[i]) == 1:
            x1_red.append(data_arr[i, 1])
            x2_red.append(data_arr[i, 2])
        else:
            x1_blue.append(data_arr[i, 1])
            x2_blue.append(data_arr[i, 2])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x1_red, x2_red, s=30, c='red', marker='s')
    ax.scatter(x1_blue, x2_blue, s=30, c='blue')

    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = (-(float)(weights[0][0]) - (float)(weights[1][0])*x1)/(float)(weights[2][0])
    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.xlabel('X2')
    plt.show()



def get_result():
    data_mat, label_mat = load_data_set()
    weights = grad_descent(data_mat, label_mat)
    print(weights)
    plot_best_fit(weights)



get_result()