import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict


PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_LEN = 50
class_names = ['yes', 'no']
LR = 2e-5
EPOCHS = 10
MODEL_NAME = 'estate_state.bin'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# x_train: [CLS] q1 [SEP] q2 [SEP]
# y_train: label

train_left = pd.read_csv('data/estate/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('data/estate/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')

# for test
# df_train = df_train[:100]
# 尝试删除q2不存在的条目查看效果
df_train['q2'] = df_train['q2'].fillna('好的')


df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=RANDOM_SEED)
df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
print(f'Train data shape: {df_train.shape}')
print(f'Test data shape: {df_test.shape}')
print(f'Validation data shape: {df_val.shape}')


class EstateDataset(Dataset):
    def __init__(self, questions, answers, targets, tokenizer, max_len):
        self.questions = questions
        self.answers = answers
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        answer = self.answers[item]
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=80,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'targets': torch.tensor(self.targets[item], dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EstateDataset(
        questions=df.q1.to_numpy(),
        answers=df.q2.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
print(data.keys())
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['token_type_ids'].shape)
print(data['targets'].shape)


class EstateModel(nn.Module):
    def __init__(self):
        super(EstateModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, len(class_names))

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output = self.dropout(pooled_output)
        return self.linear(output)

model = EstateModel()
model = model.to(device)

total_steps = len(train_data_loader) * EPOCHS

optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(model, data_loader, loss_fn,
                optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    n_examples = 0
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        targets = d["targets"].to(device)

        n_examples += d['targets'].shape[0]

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds==targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print('Train n_examples=', n_examples)
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    n_examples = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            targets = d["targets"].to(device)

            n_examples += d['targets'].shape[0]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
        print('Eval n_examples=', n_examples)
    return correct_predictions.double() / n_examples, np.mean(losses)


def train():
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device
            )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), MODEL_NAME)
            print('Saved best model to ', MODEL_NAME)
            best_accuracy = val_acc


def accuracy_on_best_model(dataloader):
    model = EstateModel()
    model.load_state_dict(torch.load(MODEL_NAME))
    model = model.to(device)
    acc, _ = eval_model(
        model=model,
        data_loader=dataloader,
        loss_fn=loss_fn,
        device=device)
    print(f'Accuracy {acc} in test dataloader!')


test_left = pd.read_csv('data/estate/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right = pd.read_csv('data/estate/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
df_no_label = test_left.merge(test_right, how='left')


df_no_label['label'] = 0
print(df_no_label.columns)

df_no_label_dataloader = create_data_loader(df_no_label, tokenizer, MAX_LEN, BATCH_SIZE)
