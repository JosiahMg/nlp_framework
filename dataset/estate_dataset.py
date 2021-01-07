import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import random
import re
import numpy as np


class EstateDataset(Dataset):
    """
    label: 判断数据是否含义标签，只有test的数据没有标签，train and val 数据都含义标签设置为True
    regularization: 是否需要数据截断操作，只需要train data进行该处理
    train_data: 是否是训练用的数据, val and test data 设置为false

    """
    def __init__(self, df, max_seq_len, bert_vocab, label=True, regularization=True, train_data=False):
        # pandas dataframe
        self.df = df
        # Q+A length
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab)
        # 数据集中是否包含预测值
        self.label = label
        self.regularization = regularization
        self.train_data = train_data
        # question: Text
        self.questions = df.q1.to_numpy()
        # answer: Text
        self.answers = df.q2.to_numpy()
        # self.tfidf = np.array([data.split(" ") for data in df.tfidf]).astype(float)

        if label:  # 有预测值的数据集
            # target: NUM [0, 1]
            self.targets = df.label.to_numpy()
        else:  # 没有预测值的数据集
            # id: NUM [0, 1, ..]
            self.ids = df.id.to_numpy()
            self.id_subs = df.id_sub.to_numpy()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        answer = self.answers[item]

        if self.regularization and self.label and self.train_data:
            if len(question) > 40:
                question = self.data_regularization(question)
            if len(answer) > 40:
                answer = self.data_regularization(answer)

        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_seq_len,
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
                # "tfidf": torch.tensor(self.tfidf[item], dtype=torch.float32),
                'targets': torch.tensor(self.targets[item], dtype=torch.float32)
            }
        else:
            id = self.ids[item]
            id_sub = self.id_subs[item]
            return {
                "input_ids": encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                # "tfidf": torch.tensor(self.tfidf[item], dtype=torch.float32),
                "id": id,
                "id_sub": id_sub
            }

    @staticmethod
    def load_data(corpus_path):

        def replace_num(text):
            text = re.sub(r"\d+", "8", text)
            text = re.sub(r"\d+-\d+", "8", text)
            text = re.sub(r"\d+.\d+.\d?", "8", text)
            text = re.sub(r"\d+.\d+", "8", text)
            text = re.sub(r"\d+", "8", text)
            return text

        train_left = pd.read_csv(corpus_path+'/'+'train.query.tsv', sep='\t', header=None)
        train_left.columns = ['id', 'q1']
        train_right = pd.read_csv(corpus_path+'/'+'train.reply.tsv', sep='\t', header=None)
        train_right.columns = ['id', 'id_sub', 'q2', 'label']
        df_train = train_left.merge(train_right, how='left')

        # 尝试删除q2不存在的条目查看效果
        df_train['q2'] = df_train['q2'].fillna('好的')

        # df_train['q1'] = df_train['q1'].apply(replace_num)
        # df_train['q2'] = df_train['q2'].apply(replace_num)

        test_left = pd.read_csv(corpus_path+'/'+'test.query.tsv', sep='\t', header=None, encoding='gbk')
        test_left.columns = ['id', 'q1']
        test_right = pd.read_csv(corpus_path+'/'+'test.reply.tsv', sep='\t', header=None, encoding='gbk')
        test_right.columns = ['id', 'id_sub', 'q2']
        df_no_label = test_left.merge(test_right, how='left')

        # df_no_label['q1'] = df_no_label['q1'].apply(replace_num)
        # df_no_label['q2'] = df_no_label['q2'].apply(replace_num)

        print("df_no_label length: ", len(df_no_label))
        return df_train, df_no_label

    @staticmethod
    def data_regularization(text):
        if random.random() < 0.9:
            split_spans = [i.span() for i in re.finditer("，|；|。|？|!", text)]
            if len(split_spans) != 0:
                span_idx = random.randint(0, len(split_spans) - 1)
                cut_position = split_spans[span_idx][1]
                if random.random() < 0.5:
                    if len(text) - cut_position > 2:
                        text = text[cut_position:]
                    else:
                        text = text[:cut_position]
                else:
                    if cut_position > 2:
                        text = text[:cut_position]
                    else:
                        text = text[cut_position:]
        return text



if __name__ == '__main__':
    corpus_dir = '../corpus/estate'
    max_seq_len = 64
    bert_vocab = 'bert-base-chinese'
    batch_size = 16
    df_train_all, df_no_label = EstateDataset.load_data(corpus_dir)
    df_train, df_val = train_test_split(df_train_all, test_size=0.2)
    print('train data shape:', df_train.shape)
    print('validation data shape:', df_val.shape)
    print('test data shape:', df_no_label.shape)
    data_train = EstateDataset(df_train, max_seq_len, bert_vocab, label=True)
    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    print('train dataloader length:', len(dataloader_train))

    data_val = EstateDataset(df_val, max_seq_len, bert_vocab, label=True)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    print('validation dataloader length:', len(dataloader_train))

    data_test = EstateDataset(df_no_label, max_seq_len, bert_vocab, label=False)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    print('test dataloader length:', len(dataloader_test))

