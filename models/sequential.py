import torch
import numpy as np
from dataset.utils_dataset import FizzBuzzDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import MultiStepLR


class FizzBuzzModel:
    def __init__(self, input_size, hidden_size, n_classes):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.n_classes)
        )
        return model


if __name__ == "__main__":
    batch_size = 64
    epochs = 5000
    dataset = FizzBuzzDataset()
    train_data, test_data = dataset.load_data()
    model = FizzBuzzModel(10, 100, 4)()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)  ## 指数衰减, 每执行一个schedulaer.step() lr*gamma
    scheduler = MultiStepLR(optimizer, milestones=[epochs*0.5, epochs*0.8], gamma=0.6)

    for epoch in range(epochs):
        for start in range(0, len(train_data[0]), batch_size):
            end = start + batch_size
            batchX = train_data[0][start:end]
            batchy = train_data[1][start:end]
            y_pred = model(batchX)
            loss = loss_fn(y_pred, batchy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss = loss_fn(model(train_data[0]), train_data[1]).item()
        print('Epoch:', epoch, 'LR: ', optimizer.state_dict()['param_groups'][0]['lr'], 'Loss:', loss)


    with torch.no_grad():
        y_pred = model(test_data[0])

    predictions = zip(range(1, 101), list(y_pred.max(dim=1)[1].data.tolist()))
    print([dataset.fizz_buzz_decode(i, x) for (i, x) in predictions])

    print(np.sum(y_pred.max(1)[1].numpy() == np.array([dataset.fizz_buzz_encode(i) for i in range(1, 101)])))
    print(y_pred.max(1)[1].numpy() == np.array([dataset.fizz_buzz_encode(i) for i in range(1, 101)]))
