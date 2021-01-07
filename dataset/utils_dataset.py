import torch
import numpy as np


"""
'hello' -> 'ohlol'
"""


class HelloDataset:
    def __init__(self):
        self.input_size = 4
        self.batch_size = 1
        self.seq_len = 5
        # vocabulary
        self.idx2char = ['e', 'h', 'l', 'o']
        x_data = [1, 0, 2, 2, 3]  # 'hello'
        y_data = [3, 1, 2, 3, 2]  # 'ohlol'

        one_hot_loopup = [[1, 0, 0, 0],  # 'e'
                          [0, 1, 0, 0],  # 'h'
                          [0, 0, 1, 0],  # 'l'
                          [0, 0, 0, 1]]  # 'o'
        x_one_hot = [one_hot_loopup[x] for x in x_data]
        self.inputs = torch.Tensor(x_one_hot).view(self.batch_size, self.seq_len, self.input_size)
        self.labels = torch.LongTensor(y_data)

    def dataloader(self):
        return [(self.inputs, self.labels)]

    def load_data(self):
        return self.inputs, self.labels


def test_hello():
    dataset = HelloDataset()
    dataloader = dataset.dataloader()
    for inputs, targets in dataloader:
        print(inputs)
        print(targets)

"""
FizzBuzz是一个简单的小游戏。游戏规则如下：
从1开始往上数数，
当遇到3的倍数的时候，说fizz，
当遇到5的倍数，说buzz，
当遇到15的倍数，就说fizzbuzz，其他情况下则正常数数。

我们可以写一个简单的小程序来决定要返回正常数值还是fizz, buzz 或者 fizzbuzz。
"""
class FizzBuzzDataset:
    def __init__(self, num_digits=10):
        self.num_digits = num_digits

    @staticmethod
    def fizz_buzz_encode(i):
        if i % 15 == 0: return 3
        elif i % 5 == 0: return 2
        elif i % 3 == 0: return 1
        else: return 0

    @staticmethod
    def fizz_buzz_decode(i, prediction):
        return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

    # 将十进制转换为二进制
    def binary_encode(self, num):
        return np.array([num >> d & 1 for d in range(self.num_digits)])

    def load_data(self):
        # 取 101- 2.pow(self.num_digits)作为训练集
        train_X = torch.Tensor([self.binary_encode(i) for i in range(101, 2**self.num_digits)])
        train_y = torch.LongTensor([self.fizz_buzz_encode(i) for i in range(101, 2**self.num_digits)])

        test_X = torch.Tensor([self.binary_encode(i) for i in range(1, 101)])
        test_y = torch.LongTensor([self.fizz_buzz_encode(i) for i in range(1, 101)])
        return (train_X, train_y), (test_X, test_y)


def test_fizz_buzz(batch_size=64):
    dataset = FizzBuzzDataset()
    train_data, test_data = dataset.load_data()
    for start in range(0, len(train_data[0]), batch_size):
        end = start + batch_size
        batchX = train_data[0][start:end]
        batchy = train_data[1][start:end]
        print(batchX.shape)
        print(batchy.shape)


if __name__ == '__main__':
    test_fizz_buzz()

