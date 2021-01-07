import numpy as np
from dataset.estate_dataset import EstateDataset
from models.bert_estate import EstateModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, GroupKFold
from utils.metrics import find_best_threshold
from utils.file_oper import delete_spec_file_subname
import configparser
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf(train_df, test_df, max_features=768):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
    train_qa = train_df['q1'] + train_df['q2']
    train_tfidf = tfidf.fit_transform(train_qa).toarray()
    # 转换成str类型并用空格拼接
    train_df['tfidf'] = np.array([" ".join(data.astype(str)) for data in train_tfidf])
    test_qa = test_df['q1'] + test_df['q2']
    test_tfidf = tfidf.transform(test_qa).toarray()
    test_df['tfidf'] = np.array([" ".join(data.astype(str)) for data in test_tfidf])
    return train_df, test_df


def create_data_loader(df, max_seq_len, bert_vocab, batch_size, label=True, train_data=False, shuffle=False):
    ds = EstateDataset(df, max_seq_len, bert_vocab, label=label, train_data=train_data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def gen_submission_file(df, valid_preds, valid_targets, test_preds, test_pred_sub_bins, k):
    threshold = find_best_threshold(valid_preds, valid_targets)
    test_pred_bins = np.average(test_preds, axis=0)
    test_pred_bins = test_pred_bins > threshold
    df['label'] = test_pred_bins.astype(int)
    df[['id', 'id_sub', 'label']].to_csv('./state_dict/submission_{}_1.csv'.format(k), index=False, header=None, sep='\t')

    label_sub = np.sum(test_pred_sub_bins, axis=0)
    label_sub = label_sub > k / 2
    df['label'] = label_sub.astype(int)
    df[['id', 'id_sub', 'label']].to_csv('./state_dict/submission_{}_2.csv'.format(k), index=False, header=None, sep='\t')


def kfold_train(k=5):
    config_ = configparser.ConfigParser()
    config_.read('./config/estate_config.ini')
    config = config_["DEFAULT"]
    corpus_dir = config['corpus_dir']
    batch_size = int(config['batch_size'])
    max_seq_len = int(config['max_seq_len'])
    # bert_vocab = config['robert_vocab']
    bert_vocab = config['bert_name']
    random_seed = int(config['random_seed'])

    df_train_all, df_no_label = EstateDataset.load_data(corpus_dir)
    # 创建tfidf 默认768维度 用于模型特征使用
    # df_train_all, df_no_label = create_tfidf(df_train_all, df_no_label)

    df_train_all = shuffle(df_train_all)

    # for test
    # df_train_all = df_train_all[:52]
    # df_no_label = df_no_label[:20]

    # df_train, df_val = train_test_split(df_train_all, test_size=0.2, random_state=random_seed)
    # train_data_loader = create_data_loader(df_train, max_seq_len, bert_name, batch_size)
    # val_data_loader = create_data_loader(df_val, max_seq_len, bert_name, batch_size)
    test_data_loader = create_data_loader(df_no_label, max_seq_len, bert_vocab, batch_size, label=False, shuffle=False)

    print(f'max_seq_len={max_seq_len}')
    print(f'bert_vocab_name={bert_vocab}')
    print(f'batch_size={batch_size}')

    gkf = GroupKFold(n_splits=k).split(X=df_train_all, groups=df_train_all.id)

    valid_preds = np.array([])
    valid_targets = np.array([])
    test_preds = []
    test_pred_sub_bins = []

    for i, (train_idx, valid_idx) in enumerate(gkf):
        print(f'-----Train k={i+1}-----')
        df_train = df_train_all.iloc[train_idx]
        df_valid = df_train_all.iloc[valid_idx]
        print(f'Train data shape: {df_train.shape}')
        print(f'Train positive data shape: {df_train[df_train["label"] == 1].shape}')
        print(f'Validation data shape: {df_valid.shape}')
        print(f'Validation positive data shape: {df_valid[df_valid["label"] == 1].shape}')


        train_data_loader = create_data_loader(df_train, max_seq_len, bert_vocab, batch_size, shuffle=True, train_data=True)
        val_data_loader = create_data_loader(df_valid, max_seq_len, bert_vocab, batch_size, shuffle=False)

        model = EstateModel()
        model.fit(train_data_loader, val_data_loader, epoch_nums=10, resume=False)

        valid_pred, _, _ = model.predict(val_data_loader)
        valid_preds = np.append(valid_preds, valid_pred)
        valid_targets = np.append(valid_targets, df_valid['label'].values)

        test_pred, test_pred_bin, _ = model.predict(test_data_loader)
        test_preds.append(test_pred)
        test_pred_sub_bins.append(test_pred_bin)

        gen_submission_file(df_no_label, valid_preds, valid_targets, test_preds, test_pred_sub_bins, i+1)

        delete_spec_file_subname(r'./state_dict/', r'epoch.')


def split_train(test_size=0.1, epoch_nums=1):
    config_ = configparser.ConfigParser()
    config_.read('./config/estate_config.ini')
    config = config_["DEFAULT"]
    corpus_dir = config['corpus_dir']
    batch_size = int(config['batch_size'])
    max_seq_len = int(config['max_seq_len'])
    # bert_vocab = config['robert_vocab']
    bert_vocab = config['bert_name']
    random_seed = int(config['random_seed'])

    df_train_all, df_no_label = EstateDataset.load_data(corpus_dir)

    # df_train_all = shuffle(df_train_all)


    df_train, df_val = train_test_split(df_train_all, test_size=test_size, random_state=random_seed)
    train_data_loader = create_data_loader(df_train, max_seq_len, bert_vocab, batch_size, shuffle=True)
    val_data_loader = create_data_loader(df_val, max_seq_len, bert_vocab, batch_size, shuffle=False)
    test_data_loader = create_data_loader(df_no_label, max_seq_len, bert_vocab, batch_size, label=False, shuffle=False)

    print(f'max_seq_len={max_seq_len}')
    print(f'bert_vocab_name={bert_vocab}')
    print(f'batch_size={batch_size}')

    print(f'Train data shape: {df_train.shape}')
    print(f'Train positive data shape: {df_train[df_train["label"] == 1].shape}')
    print(f'Validation data shape: {df_val.shape}')
    print(f'Validation positive data shape: {df_val[df_val["label"] == 1].shape}')

    model = EstateModel()
    model.fit(train_data_loader, val_data_loader, epoch_nums=epoch_nums, resume=False)

    _, all_predictions_bin, auc = model.predict(test_data_loader)
    df_no_label['label'] = all_predictions_bin
    df_no_label[['id', 'id_sub', 'label']].to_csv('./state_dict/submission_{:.4f}.csv'.format(auc), index=False, header=None, sep='\t')


def predict_to_csv():
    config_ = configparser.ConfigParser()
    config_.read('./config/estate_config.ini')
    config = config_["DEFAULT"]
    corpus_dir = config['corpus_dir']
    batch_size = int(config['batch_size'])
    max_seq_len = int(config['max_seq_len'])
    # bert_vocab = config['robert_vocab']
    bert_vocab = config['bert_name']
    random_seed = int(config['random_seed'])

    _, df_no_label = EstateDataset.load_data(corpus_dir)



    test_data_loader = create_data_loader(df_no_label, max_seq_len, bert_vocab, batch_size, label=False, shuffle=False)


    model = EstateModel()

    _, all_predictions_bin, auc = model.predict(test_data_loader)
    df_no_label['label'] = all_predictions_bin
    df_no_label[['id', 'id_sub', 'label']].to_csv('./state_dict/submission_{:.4f}.csv'.format(auc), index=False, header=None, sep='\t')


if __name__ == '__main__':
    # split_train()
    predict_to_csv()

