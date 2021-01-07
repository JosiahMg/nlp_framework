import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


x, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


def make_plot(x, y, name):
    plt.figure(figsize=(12, 8))
    plt.title(name, fontsize=30)
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])


class Layer:
    def __init__(self, n_input, n_output, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_output) * np.sqrt(1 / n_output)
        self.bias = bias if bias is not None else np.random.rand(n_output) * 0.1
        self.activation = activation  # 激活函数类型，如’sigmoid’
        self.activation_output = None  # 激活函数的输出值 o
        self.error = None  # 用于计算当前层的 delta 变量的中间变量
        self.delta = None

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.activation_output = self._apply_activation(r)
        return self.activation_output

    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        """
        计算激活函数的导数
        r: 激活函数计算后的值
        无激活函数， 导数为 1
        """
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r < 0] = 0.
            return grad
        elif self.activation == 'tanh':  # dtanh(x)/dx = 1-tanh(x)^2
            return 1 - r**2
        elif self.activation == 'sigmoid':
            return r * (1-r)
        return r


class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, x):
        for layer in self._layers:
            x = layer.activate(x)
        return x

    def backpropagation(self, x, y, learning_rate):
        # 反向传播算法实现
        # 向前计算，得到最终输出值
        output = self.feed_forward(x)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:  # 如果是最后一层
                layer.error = y - output  # 计算MSE
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:  # 隐层
                next_layer = self._layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.activation_output)

        # 循环更新权重
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i为上一层网络层的输出
            o_i = np.atleast_2d(x if i == 0 else self._layers[i-1].activation_output)
            # 梯度下降算法，delta 是公式中的负数，故这里用加号
            layer.weights += layer.delta * o_i.T * learning_rate

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        # 网络训练函数
        # one-hot 编码
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        for i in range(max_epochs):  # 训练 100 个 epoch
            for j in range(len(X_train)):  # 一次训练一个样本
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
                if i % 10 == 0:
                    # 打印出 MSE Loss
                    mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                    mses.append(mse)
                    print('Epoch: #%s, MSE: %f, Accuracy: %.2f%%' %
                          (i, float(mse), self.accuracy(self.predict(X_test), y_test.flatten()) * 100))

        return mses

    def accuracy(self, y_predict, y_test):  # 计算准确度
        return np.sum(y_predict == y_test) / len(y_test)

    def predict(self, X_predict):
        y_predict = self.feed_forward(X_predict)  # 此时的 y_predict 形状是 [600 * 2]，第二个维度表示两个输出的概率
        y_predict = np.argmax(y_predict, axis=1)
        return y_predict


nn = NeuralNetwork() # 实例化网络类
nn.add_layer(Layer(2, 25, 'sigmoid'))  # 隐藏层 1, 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid')) # 隐藏层 2, 25=>50
nn.add_layer(Layer(50, 25, 'sigmoid')) # 隐藏层 3, 50=>25
nn.add_layer(Layer(25, 2, 'sigmoid'))  # 输出层, 25=>2


nn.train(x_train, x_test, y_train, y_test, learning_rate=0.05, max_epochs=100)


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(1, -1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predic = model.predict(X_new)
    zz = y_predic.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF590', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)


plt.figure(figsize=(12, 8))
plot_decision_boundary(nn, [-2, 2.5, -1, 2])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()


y_predict = nn.predict(x_test)
print(nn.accuracy(y_predict, y_test.flatten()))
