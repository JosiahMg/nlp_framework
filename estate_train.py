from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import os
import configparser
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from models.bert_estate import EstateModel
from dataset.estate_dataset import EstateDataset
from utils.metrics import find_best_threshold



if __name__ == '__main__':
    def create_data_loader(df, max_seq_len, bert_name, batch_size, label=True):
        ds = EstateDataset(df, max_seq_len, bert_name, label=label)
        return DataLoader(ds, batch_size=batch_size)

    config_ = configparser.ConfigParser()
    config_.read('./config/estate_config.ini')
    config = config_["DEFAULT"]
    corpus_dir = config['corpus_dir']
    batch_size = int(config['batch_size'])
    max_seq_len = int(config['max_seq_len'])
    bert_name = config['bert_name']
    random_seed = int(config['random_seed'])
    df_train, df_no_label = EstateDataset.load_data(corpus_dir)

    # for test
    df_train = df_train[:100]

    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)
    train_data_loader = create_data_loader(df_train, max_seq_len, bert_name, batch_size)
    val_data_loader = create_data_loader(df_val, max_seq_len, bert_name, batch_size)

    trainer = EstateTrainer(train_data_loader, val_data_loader)
    dynamic_lr = trainer.lr
    start_epoch = 1
    train_epoches = 9999
    all_auc = []
    threshold = 999
    patient = 5
    best_loss = 999999999
    for epoch in range(start_epoch, start_epoch + train_epoches):
        print("train with learning rate {}".format(str(dynamic_lr)))
        # 训练一个epoch
        trainer.train(epoch)
        auc = trainer.test(epoch)
        all_auc.append(auc)
        best_auc = max(all_auc)

        if all_auc[-1] < best_auc:
            threshold += 1
            dynamic_lr *= 0.8
            trainer.init_optimizer(lr=dynamic_lr)
        else:
            trainer.save_state_dict(trainer.bert_model, epoch,
                                    state_dict_dir=trainer.config["state_dict_dir"],
                                    file_path="estate.model")
            threshold = 0

        if threshold >= patient:
            # 保存当前epoch模型参数
            print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_auc))))
            print("early stop!")
            break

