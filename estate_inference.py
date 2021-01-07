import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

BERT_BASE_ZH_MODEL = 'bert-base-chinese'
PRE_TRAINED_MODEL_PATH = 'pretrained_model/'
ROBERT_WWN_L_ZH = PRE_TRAINED_MODEL_PATH + 'chinese_roberta_wwm_large_ext_pt/'
ROBERT_WWN_L_ZH_VOCAB = ROBERT_WWN_L_ZH + 'vocab.txt'
ROBERT_WWN_L_ZH_CONF = ROBERT_WWN_L_ZH + 'bert_config.json'
ROBERT_WWN_L_ZH_MODEL = ROBERT_WWN_L_ZH + 'pytorch_model.bin'



RANDOM_SEED = 42
BATCH_SIZE = 64
MAX_LEN = 128
class_names = ['yes', 'no']
LR = 2e-5
EPOCHS = 5
MODEL_NAME = 'estate_state.bin'

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# x_train: [CLS] q1 [SEP] q2 [SEP]
# y_train: label

train_left = pd.read_csv('data/estate/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('data/estate/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')

# for test
df_train = df_train[:100]
# 尝试删除q2不存在的条目查看效果
df_train['q2'] = df_train['q2'].fillna('好的')


# df_train, df_test = train_test_split(df_train, test_size=0.05, random_state=RANDOM_SEED)
# df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
# df_val = df_test
# print(f'Train data shape: {df_train.shape}')
# print(f'Test data shape: {df_test.shape}')
# print(f'Validation data shape: {df_val.shape}')


class EstateDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label=True):
        self.label = label
        if label:
            self.targets = df.label.to_numpy()
        else:
            self.ids = df.id.to_numpy()
            self.id_subs = df.id_sub.to_numpy()
        self.questions = df.q1.to_numpy()
        self.answers = df.q2.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        answer = self.answers[item]
        if not self.label:
            id = self.ids[item]
            id_sub = self.id_subs[item]
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if self.label:
            return {
                "input_ids": encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'targets': torch.tensor(self.targets[item], dtype=torch.float32)
                }
        else:
            return {
                "input_ids": encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                "id": id,
                "id_sub": id_sub
            }


# tokenizer = BertTokenizer.from_pretrained(ROBERT_WWN_L_ZH_VOCAB)
tokenizer = BertTokenizer.from_pretrained(BERT_BASE_ZH_MODEL)


def create_data_loader(df, tokenizer, max_len, batch_size, label=True):
    ds = EstateDataset(
        df,
        tokenizer=tokenizer,
        max_len=max_len,
        label=label
    )
    return DataLoader(ds, batch_size=batch_size)


# train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
# val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
# test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
#
# data = next(iter(train_data_loader))
# print(data.keys())
# print(data['input_ids'].shape)
# print(data['attention_mask'].shape)
# print(data['token_type_ids'].shape)
# print(data['targets'].shape)


class EstateModel(nn.Module):
    def __init__(self):
        super(EstateModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_BASE_ZH_MODEL)
        # self.config = BertConfig.from_pretrained(ROBERT_WWN_L_ZH_CONF)
        # self.bert = BertModel.from_pretrained(ROBERT_WWN_L_ZH_MODEL, config=self.config)

        # self.bert fixed
        # for p in self.parameters():
        #     p.requires_grad = False

        self.dropout = nn.Dropout(p=0.2)
        # self.linear = nn.Linear(self.bert.config.hidden_size*4, len(class_names))  # for CrossEntropyLoss() loss funtion
        self.linear = nn.Linear(self.bert.config.hidden_size*4, 1) # for BCELoss() function

    def forward(self, input_ids, attention_mask, token_type_ids):

        # last_hidden_state.shape: (batch, seq_len, hidden_size)
        # pooled_output.shape: (batch, hidden_size)
        last_hidden_state, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # output = self.dropout(pooled_output)
        pooled_avg = torch.mean(last_hidden_state, dim=1)  # (batch, hidden_size)
        pooled_max, _ = torch.max(last_hidden_state, dim=1)  # (batch, hidden_size)
        last_word = last_hidden_state[:, -1]  # (batch, hidden_size)
        first_word = last_hidden_state[:, 0]  # (batch, hidden_size)
        output = torch.cat((pooled_avg, pooled_max, last_word, first_word), dim=1)  # shape:(batch, 4*hidden_size)
        output = self.dropout(output)
        output = self.linear(output)
        return torch.sigmoid(output)

# model = EstateModel()
# model = model.to(device)
#
# # total_steps = len(train_data_loader) * EPOCHS
# total_steps = 1  # error, just for build correct
# optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=total_steps
#     )
# loss_fn_m = nn.CrossEntropyLoss().to(device)
# loss_fn_b = nn.BCELoss().to(device)


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

        outputs = outputs.squeeze() # for BCELoss()
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        # _, preds = torch.max(outputs, dim=1)  # for CrossEntoryLoss()
        preds = outputs.ge(0.5).float()  # for BCELoss()
        correct_predictions += torch.sum(preds==targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step()
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

            outputs = outputs.squeeze()  # for BCELoss()
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            # _, preds = torch.max(outputs, dim=1)  # for CrossEntropyLoss()
            # preds = torch.ceil(outputs)  # for CrossEntropyLoss()
            preds = outputs.ge(0.5).float()
            correct_predictions += torch.sum(preds == targets)
        print('Eval n_examples=', n_examples)
    return correct_predictions.double() / n_examples, np.mean(losses)


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer):
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
            device
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


# def accuracy_on_best_model(dataloader):
#     model = EstateModel()
#     model.load_state_dict(torch.load(MODEL_NAME))
#     model = model.to(device)
#     acc, _ = eval_model(
#         model=model,
#         data_loader=dataloader,
#         loss_fn=loss_fn_b,
#         device=device)
#     print(f'Accuracy {acc} in test dataloader!')


test_left = pd.read_csv('data/estate/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right = pd.read_csv('data/estate/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
df_no_label = test_left.merge(test_right, how='left')
print("df_no_label length: ", len(df_no_label))
# for test
# df_no_label = df_no_label[:200]

print(df_no_label.columns)

df_no_label_dataloader = create_data_loader(df_no_label, tokenizer, MAX_LEN, BATCH_SIZE, label=False)
data = next(iter(df_no_label_dataloader))
print(data.keys())
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['token_type_ids'].shape)
print(data['id'].shape)
print(data['id_sub'].shape)


def get_predictions(model, dataloader):
    model = model.eval()
    ids = []
    id_subs = []
    predictions = []
    count = 0
    with torch.no_grad():
        for d in dataloader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            id = d['id']
            id_sub = d['id_sub']

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            # _, preds = torch.max(outputs, dim=1)
            preds = outputs.ge(0.5).float()
            ids.extend(id)
            id_subs.extend(id_sub)
            predictions.extend(preds)

    ids = torch.stack(ids).numpy()
    id_subs = torch.stack(id_subs).numpy()
    predictions = torch.stack(predictions).squeeze().cpu().numpy().astype(np.int)

    return ids, id_subs, predictions


def show_token_count(data, ques=True):
  token_lens = []
  if ques:
    tmp = data.q1
  else:
    tmp = data.q2

  for txt in tmp:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

  sns.distplot(token_lens)
  plt.xlim([0, 80]);
  plt.xlabel('Token count')


def search_by_f1(y_true, y_pred):
    best_score = 0
    best_thres = 0
    for i in range(30, 60):
        thres = i / 100
        y_pred_bin = (y_pred > thres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best_score:
            best_score = score
            best_thres = thres
    print('best F1-score: ', best_score)
    print('threshold: ', best_thres)
    return best_score, best_thres


def k_cross_validattion(df, k=5):
    model = EstateModel()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    loss_fn_b = nn.BCELoss().to(device)

    gkf = GroupKFold(n_splits=k).split(X=df, groups=df.id)
    k = 0
    for train_idx, valid_idx in gkf:
        k += 1
        print(f'-----Train k={k}-----')
        df_train = df.loc[train_idx]
        df_valid = df.loc[valid_idx]
        print(f'Train data shape: {df_train.shape}')
        print(f'Validation data shape: {df_valid.shape}')

        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        val_data_loader = create_data_loader(df_valid, tokenizer, MAX_LEN, BATCH_SIZE)
        train(model, train_data_loader, val_data_loader, loss_fn_b, optimizer)







def generial_csv():
    model = EstateModel()
    model.load_state_dict(torch.load(MODEL_NAME))
    model = model.to(device)
    ids, id_subs, preds = get_predictions(model, df_no_label_dataloader)

    df_label_data = pd.DataFrame({'id': ids, 'id_sub': id_subs, 'label': preds})
    print(df_label_data.head())
    df_label_data[['id', 'id_sub', 'label']].to_csv('submission.csv', index=False, header=None, sep='\t')


k_cross_validattion(df_train)
generial_csv()
