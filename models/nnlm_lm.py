"""
基于论文 A Neural Probabilistic Language Model 实现的语言模型

"""
import torch
import torch.nn as nn
import tqdm
import numpy as np
"""
model : 
    y1 = b + xw
    y2 = tanh(d + xH)
    y = y1 + U * y2
计算流程:
    1. 首先将输入的n-1个单词转换为词向量(nn.embedding),然后将这n-1个单词进行concat, 此时的维度为(n-1)*embed_size, 用X表示
    2. 将x送入到隐层  y2 = tanh(d+x*H)
    3. 输出层用V个节点，每个节点表示输出对应单词的概率,最终模型的输出 y = b + x*w + y2*U
每个参数的shape如下：
x.shape: (batch_size, embed_size*seq_len)
w.shape: (embed_size*seq_len, vocab_size)
b.shape: (vocab_size, )
H.shape: (embed_size*seq_len, hidden_size)
d.shape: (hidden_size, )
U.shape: (hidden_size, vocab_size)

"""


class NNLMParamModelBase(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, hidden_size):
        super(NNLMParamModelBase, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.W = nn.Parameter(torch.randn(seq_len*embed_size, vocab_size))
        self.b = nn.Parameter(torch.randn(vocab_size))
        self.H = nn.Parameter(torch.randn(seq_len*embed_size, hidden_size))
        self.d = nn.Parameter(torch.randn(hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, vocab_size))

    def forward(self, x):  # (batch, seq_len)
        x = self.emb(x)    # (batch, seq_len, emb_size)
        x = x.view(-1, self.embed_size*self.seq_len)
        y1 = self.b + torch.mm(x, self.W)
        y2 = self.d + torch.mm(x, self.H)
        outputs = y1 + torch.mm(y2, self.U)
        return outputs


class NNLMLinearModelBase(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, hidden_size):
        super(NNLMLinearModelBase, self).__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(seq_len*embed_size, vocab_size)
        self.linear2 = nn.Linear(seq_len*embed_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):  # (batchSize=5, seqLen=2)
        x = self.emb(x)  # x.shape: (batch, seq, embsize)
        y1 = self.linear1(x.view(-1, self.embed_size*self.seq_len))  # y.shape:(batch, vocab)
        y2 = torch.tanh(self.linear2(x.view(-1, self.embed_size*self.seq_len)))  # y.shape:(batch, hidden_size)
        outputs = y1 + self.linear3(y2)  # outputs.shape: (batch, vocab)
        return outputs


class NNLMModel:
    def __init__(self, vocab_size, embed_size, seq_len, hidden_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')

        self.model = NNLMLinearModelBase(vocab_size, embed_size, seq_len, hidden_size).to(self.device)
        # self.model = NNLMParamModelBase(vocab_size, embed_size, seq_len, hidden_size).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # 每调用一次，lr变成原来的0.7倍
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.7)

    def fit(self, train_data_loader, val_data_loader, epoch_nums=50, patient=3):
        start_epoch = 1
        best_loss = 99999.
        threshold = 0
        loss_list = []
        for epoch in range(start_epoch, epoch_nums+1):
            self.train(epoch, train_data_loader)

            loss = self.test(epoch, val_data_loader)
            loss_list.append(loss)

            if best_loss > loss:
                best_loss = loss
                threshold = 0
            else:
                self.scheduler.step()
                threshold += 1

            if threshold >= patient:
                # 保存当前epoch模型参数
                print("epoch {} has the lowest loss".format(start_epoch+np.argmin(loss_list)))
                print("early stop!")
                break


    def predict(self, x):
        x = x.to(self.device)
        return self.model(x).argmax(dim=1)

    def train(self, epoch, train_data_loader):
        # 一个epoch的train
        self.model.train()
        self.iteration(epoch, train_data_loader, train=True)

    """
    验证集测试
    """
    def test(self, epoch, val_data_loader):
        # 一个epoch的validation, return auc in test data
        self.model.eval()
        with torch.no_grad():
            return self.iteration(epoch, val_data_loader, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        for i, (inputs, targets) in data_iter:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs) # (batch, vocab_size)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            data_iter.write(f'{str_code}: In epoch:{epoch}, loss = {total_loss/(i+1)}')

        if not train:
            return total_loss/len(data_iter)

