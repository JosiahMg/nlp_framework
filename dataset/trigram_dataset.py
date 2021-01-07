import torch
from torch.utils.data import DataLoader, TensorDataset

"""
构造简单的bi-gram 语言模型
"""


class TriGramDataset:
    def __init__(self):
        self.sentence1 = 'i like dog'
        self.sentence2 = 'i hate coffee'
        self.sentence3 = 'i love milk'
        self.sentence4 = 'i am study'
        self.sentence5 = 'i miss you'
        word_list = []
        for i in range(1, 6):
            sent = eval('self.sentence' + str(i))
            word_list.append(sent)
        word_list = ' '.join(word_list).split()
        self.vocab = list(set(word_list))
        self.seq_len = 2
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for i, word in enumerate(self.vocab)}

    def load_data(self):
        inputs = []
        targets = []
        for i in range(1, 6):
            sent = eval('self.sentence' + str(i))
            inputs.append([self.word2idx[w] for w in sent.split()[:-1]])
            targets.append(self.word2idx[sent.split()[-1]])

        inputs = torch.LongTensor(inputs)
        targets = torch.LongTensor(targets)
        return inputs, targets

    def dataloader(self, batch_size=5, shuffle=False, drop_last=False):
        inputs, targets = self.load_data()
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def get_vocab_size(self):
        return len(self.vocab)

    def get_seq_len(self):
        return self.seq_len


if __name__ == '__main__':
    dataset = TriGramDataset()
    dataloader = dataset.dataloader()

    for inputs, targets in dataloader:
        for i in range(5):
            for j in range(2):
                print(dataset.idx2word[inputs[i][j].item()], end=' ')
            print(dataset.idx2word[targets[i].item()])

