import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_LEN = 50
class_names = ['yes', 'no']

# x_train: [CLS] q1 [SEP] q2 [SEP]
# y_train: label

train_left = pd.read_csv('data/estate/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('data/estate/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')

# 尝试删除q2不存在的条目查看效果
df_train['q2'] = df_train['q2'].fillna('好的')


# test_left = pd.read_csv('data/estate/test.query.tsv',sep='\t',header=None, encoding='gbk')
# test_left.columns = ['id','q1']
# test_right = pd.read_csv('data/estate/test.reply.tsv',sep='\t',header=None, encoding='gbk')
# test_right.columns=['id','id_sub','q2']
# df_no_label = test_left.merge(test_right, how='left')

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
        questions=df.q1,
        answers=df.q2,
        targets=df.label,
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

