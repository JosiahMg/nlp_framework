# -*- coding: utf-8 -*-
# file: bert_pretrained.py
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

import transformers
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from config.apps_reviews_config import *
import numpy as np
from preprocess.apps_reviews import ReviewDataloader
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

"""
model:
    bert -> pooled_output -> dropout -> linear(n_classes)
"""


class BertPreTrainedClassifier(nn.Module):
    """
    input_data: apps_reviews.csv
    x_data:
    """
    def __init__(self, n_classes):
        super(BertPreTrainedClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        print('Model init success!')
        print('  - Model name: ', PRE_TRAINED_MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask = attention_mask
        )
        output = self.drop(pooled_output)
        return self.linear(output)


class SentimentClassifierAppsReviews:
    """
    Sentiment classifier for apps_reviews.csv
    """
    def __init__(self,
                data_loader_all,  # dict{"train", "test", "validation"}
                device=device):
        self.train_data_loader = data_loader_all["train"]
        self.test_data_loader = data_loader_all["test"]
        self.val_data_loader = data_loader_all["validation"]
        self.total_steps = len(self.train_data_loader)*EPOCHS
        self.device = device
        self.model = BertPreTrainedClassifier(len(class_names)).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=LR, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps
        )
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        print('Init model environment success!')
        print('  - learning rate: ', LR)
        print('  - device: ', device)
        print('  - EPOCHS: ', EPOCHS)

    @staticmethod
    def train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler):
        model = model.train()
        losses = []
        correct_predictions = 0
        n_examples = 0

        for d in train_data_loader:
            n_examples += d["input_ids"].shape[0]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)

    @staticmethod
    def eval_model(model, val_data_loader, loss_fn, device):
        model = model.eval()
        losses = []
        correct_predictions = 0
        n_examples = 0

        with torch.no_grad():
            for d in val_data_loader:
                n_examples += d["input_ids"].shape[0]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    def train(self):
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            train_acc, train_loss = self.train_epoch(
                model=self.model,
                train_data_loader=self.train_data_loader,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=self.scheduler
            )
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(
                model=self.model,
                val_data_loader=self.val_data_loader,
                loss_fn=self.loss_fn,
                device=self.device
            )
            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), MODEL_NAME)
                best_accuracy = val_acc
                print('Saved best model to ', MODEL_NAME)

        acc, _ = self.eval_model(
            model=self.model,
            val_data_loader=self.test_data_loader,
            loss_fn=self.loss_fn,
            device=self.device
        )
        print(f'Accuracy {acc} in test dataloader!')
        print('Complete the training!')
        return history

    @staticmethod
    def load_state_model():
        model=BertPreTrainedClassifier(len(class_names))
        model.load_state_dict(torch.load(MODEL_NAME))
        model = model.to(device)
        return model

    def get_model(self):
        return self.model

    @staticmethod
    def get_predictions(model, data_loader):
        model = model.eval()
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                texts = d["review_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


if __name__ == '__main__':
    loadtext = ReviewDataloader()
    dataloader = loadtext.get_data_loader()
    trainer = SentimentClassifierAppsReviews(dataloader)

    trainer.train()

    model = trainer.get_model()
    y_review_texts, y_pred, y_pred_probs, y_test = trainer.get_predictions(model, dataloader['test'])
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))