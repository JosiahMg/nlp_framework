import torch
import torch.nn as nn
import tqdm


class RnnModelBase(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super(RnnModelBase, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers,
                                batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, input):
        batch_size = input.size(0)
        hidden = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        output, hn = self.rnn(input, hidden)
        output = output.view(-1, self.hidden_size*self.num_directions)
        return output


class RnnModel:
    def __init__(self):
        input_size = 4
        hidden_size = 4
        num_layers = 1

        self.model = RnnModelBase(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=False)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def fit(self, train_data_loader, tese_data_loader, epoch_nums=50):
        start_epoch = 1
        loss_list = []
        for epoch in range(start_epoch, epoch_nums+1):
            self.train(epoch, train_data_loader)
            loss = self.test(epoch, tese_data_loader)
            loss_list.append(loss)

        return loss_list

    def train(self, epoch, dataloader):
        self.model.train()
        self.iteration(epoch, dataloader, train=True)

    def test(self, epoch, dataloader):
        self.model.eval()
        with torch.no_grad():
            return self.iteration(epoch, dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):

        str_code = 'train' if train else 'test'

        total_loss = 0

        for i, (inputs, targets) in enumerate(dataloader):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'{str_code}: In epoch:{epoch}, loss = {total_loss / (i + 1)}')

            if not train:
                return total_loss / len(dataloader)

    def predict(self, x):
        return self.model(x).argmax(dim=1)


if __name__ == '__main__':
    from dataset.utils_dataset import HelloDataset

    model = RnnModel()
    dataset = HelloDataset()
    dataloader = dataset.dataloader()
    model.fit(dataloader, dataloader)
    outputs = model.predict(dataset.dataset()[0])
    print('Predicted: ', ''.join([dataset.idx2char[x] for x in outputs]), end='')

