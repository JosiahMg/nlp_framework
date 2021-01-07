from transformers import BertModel, BertConfig
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
from dataset.estate_dataset import EstateDataset
from utils.metrics import find_best_threshold


"""使用最后一层的mean以及max进行cat"""


class BertModelBase(nn.Module):
    def __init__(self, bert_config, bert_name, dropout, bidirectional=False):
        super(BertModelBase, self).__init__()
        if bert_config is None:
            self.bert = BertModel.from_pretrained(bert_name)
        else:
            self.config = BertConfig.from_pretrained(bert_config)
            self.bert = BertModel.from_pretrained(bert_name, config=self.config)

        # self.bert fixed
        # for p in self.parameters():
        #     p.requires_grad = False
        num_direction = 2 if bidirectional else 1

        hidden_size = self.bert.config.hidden_size

        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=1, bidirectional=bidirectional, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)
        # self.linear = nn.Linear(self.bert.config.hidden_size*4, len(class_names))  # for CrossEntropyLoss() loss funtion
        self.linear = nn.Linear(num_direction*hidden_size*4, 1) # for BCELoss() function

    def forward(self, input_ids, attention_mask, token_type_ids):

        # last_hidden_state.shape: (batch, seq_len, hidden_size)
        # pooled_output.shape: (batch, hidden_size)
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state, _ = self.gru(last_hidden_state)   # (batch, seq, hidden_size*bidirection)
        pooled_avg = torch.mean(last_hidden_state, dim=1)  # (batch, hidden_size*bidirection)
        pooled_max, _ = torch.max(last_hidden_state, dim=1)  # (batch, hidden_size*bidirection)
        last_word = last_hidden_state[:, -1]  # (batch, hidden_size*bidirection)
        first_word = last_hidden_state[:, 0]  # (batch, hidden_size*bidirection)
        output = torch.cat((pooled_avg, pooled_max, last_word, first_word), dim=1)  # shape:(batch, 4*hidden_size*bidirection)
        # output = torch.cat((pooled_avg, pooled_max), dim=1)  # shape:(batch, 2*hidden_size)
        output = self.dropout(output)
        output = self.linear(output)
        return torch.sigmoid(output)


class EstateModel:
    def __init__(self):

        config_ = configparser.ConfigParser()
        config_.read('./config/estate_config.ini')
        self.config = config_["DEFAULT"]

        self.state_dict_dir = self.config['state_dict_dir']
        # bert-base-chinese
        # bert_name = self.config['bert_name']

        # chinese roberta wwm large
        # bert_config = self.config['robert_config']
        # bert_name = self.config['robert_name']
        bert_config = None
        bert_name = self.config['bert_name']

        dropout_rate = float(self.config['dropout_rate'])
        corpus_path = self.config['corpus_dir']
        max_seq_len = int(self.config['max_seq_len'])
        self.lr = float(self.config['lr'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device("cpu")

        if torch.cuda.is_available():
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')

        self.bert_model = BertModelBase(bert_config, bert_name, dropout_rate).to(self.device)
        # 声明需要优化的参数, 并传入Adam优化器
        self.optim_parameters = list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)
        self.loss_fn = nn.BCELoss().to(self.device)

        if not os.path.exists(self.config["state_dict_dir"]):
            os.mkdir(self.config["state_dict_dir"])

        print(f'bert_name={bert_name}')


    def init_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # self.optimizer = torch.optim.SGD(self.optim_parameters, lr=lr, momentum=0.9, weight_decay=1e-2)

    # 判断是否有保存的模型
    def check_checkpoint(self, path_dir='./state_dict'):
        if os.path.exists(path_dir):
            dic_lis = [i for i in os.listdir(path_dir)]
            dic_lis = [i for i in dic_lis if r"model.epoch." in i]
            if len(dic_lis) != 0:
                return True
        return False

    """
    多次epoch训练
    epoch_nums: 训练轮次数
    patient: 设置训练时当前评价比之前差，连续patient次则退出训练
    resume: 断点续训
    """
    def fit(self, train_data_loader, val_data_loader, epoch_nums=999, patient=3, resume=True):
        all_auc = []
        if resume and self.check_checkpoint(self.config['state_dict_dir']):
            start_epoch, _, auc = self.load_model()
            start_epoch += 1
            all_auc.append(auc)
            print(f'Load checkpoint to continue training: start_epoch: {start_epoch}, lr: {self.lr}, auc: {auc}')
        else:
            start_epoch = 1

        dynamic_lr = self.lr
        threshold = 0
        for epoch in range(start_epoch, epoch_nums+1):
            print("train with learning rate {}".format(str(dynamic_lr)))
            self.train(epoch, train_data_loader)
            auc, thres = self.test(epoch, val_data_loader)
            all_auc.append(auc)
            best_auc = max(all_auc)

            if all_auc[-1] < best_auc:
                threshold += 1
                dynamic_lr *= 0.5
                self.init_optimizer(lr=dynamic_lr)
            else:
                self.save_state_dict(self.bert_model, self.optimizer, dynamic_lr, epoch, thres, best_auc,
                                     state_dict_dir=self.config["state_dict_dir"], file_path="estate.model")
                threshold = 0

            if threshold >= patient:
                # 保存当前epoch模型参数
                print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_auc))))
                print("early stop!")
                break


    """
    单次epoch的训练，使用auc指标进行训练集的测试
    epoch: 第几轮训练
    train_data_loader: 训练数据
    """
    def train(self, epoch, train_data_loader):
        # 一个epoch的train
        self.bert_model.train()
        self.iteration(epoch, train_data_loader, train=True)

    """
    验证集测试
    """
    def test(self, epoch, val_data_loader):
        # 一个epoch的validation, return auc in test data
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, val_data_loader, train=False)

    def iteration(self, epoch, data_loader, train=True, df_name="df_log.pickle"):
        # 初始化一个pandas DataFrame进行训练日志的存储
        df_path = self.config["state_dict_dir"] + "/" + df_name
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "test_loss", "test_auc"])
            df.to_pickle(df_path)
            print("log DataFrame created!")

        # 进度条显示
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        total_loss = 0
        # 存储所有预测的结果和标记, 用来计算auc
        all_predictions, all_labels = [], []

        for i, data in data_iter:
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            token_type_ids = data['token_type_ids'].to(self.device)
            # tfidf = data['tfidf'].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.bert_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)
            loss = self.loss_fn(outputs.squeeze(1), targets)

            predictions = outputs.detach().cpu().numpy().reshape(-1).tolist()
            labels = targets.cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            # Compute auc
            fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                                                     y_score=all_predictions)
            auc = metrics.auc(fpr, tpr)

            # 反向传播
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.optim_parameters, max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()

            if train:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": total_loss / (i + 1), "train_auc": auc,
                    "test_loss": 0, "test_auc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": 0, "train_auc": 0,
                    "test_loss": total_loss / (i + 1), "test_auc": auc
                }
            # 打印日志信息
            if (i+1) % 10 == 0 or (i+1) == len(data_iter):
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        threshold_ = find_best_threshold(all_predictions, all_labels)
        print(str_code + " best threshold: " + str(threshold_))

        # 将当前epoch的情况记录到DataFrame里
        if train:
            df = pd.read_pickle(df_path)
            df = df.append([log_dic])
            df.reset_index(inplace=True, drop=True)
            df.to_pickle(df_path)
        else:
            # 去除value为0和key为epoch的选项
            log_dic = {k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dic.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            # 返回auc, 作为early stop的衡量标准
            return auc, threshold_

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        dic_lis = [i for i in dic_lis if r"model.epoch." in i]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, optimizer, lr, epoch, thres, auc,
                        state_dict_dir="./state_dict", file_path="bert.model"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + file_path + ".epoch.{}".format(str(epoch))
        model.to("cpu")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            "thres": thres,
            "auc": auc,
            'epoch': epoch
        }
        torch.save(checkpoint, save_path)
        print("{} saved in epoch:{}".format(save_path, epoch))
        model.to(self.device)

    def load_model(self, dir_path='./state_dict'):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.bert_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        thres = checkpoint['thres']
        self.lr = checkpoint['lr']
        auc = checkpoint['auc']
        torch.cuda.empty_cache()
        self.bert_model.to(self.device)
        print(f"{checkpoint_dir} loaded, epoch={start_epoch}, lr={self.lr}")
        return start_epoch, thres, auc

    def predict(self, data_loader, threshold=None):
        self.bert_model.eval()
        start_epoch, thres, auc = self.load_model()
        if threshold:
            thres = threshold
        print(f'Load checkpoint at epoch: {start_epoch} thres: {thres} auc: {auc} to predict.')
        all_predictions = []
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Predicting",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        with torch.no_grad():
            for i, data in data_iter:
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                # tfidf = data['tfidf'].to(self.device)

                outputs = self.bert_model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)

                predictions = outputs.detach().cpu().numpy().reshape(-1).tolist()
                all_predictions.extend(predictions)

        all_predictions_bin = np.array(all_predictions) > thres
        all_predictions_bin = all_predictions_bin.astype(int)

        # all_predictions为实际预测值(0~1)
        # all_predictions_bin 为0或1
        return np.array(all_predictions), all_predictions_bin, auc
