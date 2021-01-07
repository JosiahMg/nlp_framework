# -*- coding: utf-8 -*-
# file: app_review_dataset.py
# author: JosiahMg
# time: 11/02/2020 4:51 PM
# Copyright 2020 JosiahMg. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import transformers
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from transformers import BertTokenizer
from config.apps_reviews_config import *

"""
filename: 
    apps_reviews.csv
format:
    userName	userImage	content	score	thumbsUpCount	reviewCreatedVersion	at	replyContent	repliedAt	sortOrder	appId
useful:
    content(x_str)     score(y_int)
"""

class LoadReviewCsv:
    """
    split data into [train, validation, test]
    each data split ['x_data', 'y_data']
    """
    def __init__(self):
        self.df = pd.read_csv(PATH)
        self.df['sentiment'] = self.df.score.apply(self.to_sentiment)
        self.class_names = ['negative', 'neutral', 'positive']
        self.df_train, df_test = train_test_split(self.df, test_size=0.1, random_state=RANDOM_SEED)
        self.df_val, self.df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
        print('Load apps_reviews.csv success!')
        print('  - Train data shape: ', self.df_train.shape)
        print('  - Test data shape: ', self.df_test.shape)
        print('  - Validation data shape: ', self.df_val.shape)


    """
    return data format: 
        {
            "train": {"x_data": data, "y_data": data},
            "test": {"x_data": data, "y_data": data},
            "validation": {"x_data": data, "y_data": data}
        }
    """

    def get_data(self):
        return {
            'train': {
                "x_data": self.df_train.content.to_numpy(),
                "y_data": self.df_train.sentiment.to_numpy()
            },
            'test': {
                "x_data": self.df_test.content.to_numpy(),
                "y_data": self.df_test.sentiment.to_numpy()
            },
            "validation": {
                "x_data": self.df_val['content'].to_numpy(),
                "y_data": self.df_val['sentiment'].to_numpy()
            }
        }

    # score(1-5) -> sentiment(0-2)
    def to_sentiment(self, rating):
        rating = int(rating)
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else:
            return 2

    # plot counter figure
    def plt_dist(self, targets, xticklabel=None):
        ax = sns.countplot(targets)
        if xticklabel:
            ax.set_xticklabels(xticklabel)
        plt.show()


class ReviewDataset(Dataset):
    """
    create dataset
    reviews: x_data
    targets: y_data
    """
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


"""
    retrun DataLoader
    ['train', 'test', 'validation']
"""
class ReviewDataloader():
    def __init__(self,
                 load_data=LoadReviewCsv,
                 review_dataset=ReviewDataset,
                 tokenizer=BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME),
                 max_len=MAX_LEN,
                 batch_size=BATCH_SIZE):

        self.load_data = load_data()
        self.review_dataset = review_dataset
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        print("Dataloader init...")
        print(f"  - batch_size={batch_size}")
        print(f"  - seq_len={max_len}")
        print(f"  - tokenizer={PRE_TRAINED_MODEL_NAME}")

    def get_data_loader(self):
        text_data = self.load_data.get_data()
        data_loader = {}
        for it in ['train', 'test', 'validation']:
            dataset = self.review_dataset(
                reviews=text_data[it]["x_data"],
                targets=text_data[it]["y_data"],
                tokenizer=self.tokenizer,
                max_len=self.max_len
            )
            data_loader[it] = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

def get_apps_reviews():
    loadtext = ReviewDataloader()
    dataloader = loadtext.get_data_loader()
    return dataloader


if __name__ == '__main__':
    loadtext = ReviewDataloader()
    dataloader = loadtext.get_data_loader()
    print(dataloader.keys())
    n = 0
    for d in dataloader['validation']:
        n += d['input_ids'].shape[0]
        x_data = d["input_ids"].to(device)  # shape: (batchSize, seqLen)
        x_data_mask = d["attention_mask"].to(device)  # shape: (batchSize, seqLen)
        y_data = d["targets"].to(device)  # shape: (batchSize)

    print('number: ', n)


