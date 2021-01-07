from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from dataset.tianchi_news_dataset import NewsDataset
import pandas as pd
import fasttext
import os


class CountModel:
    def __init__(self, max_features=3000, ngram_range=(1, 1)):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = RidgeClassifier()

    def fit(self, train_df):
        train_data = self.vectorizer.fit_transform(train_df['text']).toarray()
        self.model.fit(train_data, train_df.label.values)

    def predict(self, test_df):
        test_data = self.vectorizer.transform(test_df['text']).toarray()
        val_pred = self.model.predict(test_data)
        # score = f1_score(test_data['label'].values, val_pred, average='macro')
        return val_pred



class TfidfModel:
    def __init__(self, max_features=3000, ngram_range=(1, 1)):
        self.tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self.model = RidgeClassifier()

    def fit(self, train_df):
        train_data = self.tfidf.fit_transform(train_df['text']).toarray()
        self.model.fit(train_data, train_df.label.values)

    def predict(self, test_df):
        test_data = self.tfidf.transform(test_df['text']).toarray()
        val_pred = self.model.predict(test_data)
        # score = f1_score(test_data['label'].values, val_pred, average='macro')
        return val_pred


def test_tfidf_model():
    train_df, test_df = NewsDataset().load_data()
    model = CountModel()
    model.fit(train_df)
    df = pd.DataFrame()
    df['label'] = model.predict(test_df)
    df.to_csv('submit_{}.csv'.format('Count'), index=None)


class FasttextModel:
    # loss: loss function {ns, hs, softmax} [softmax]
    # wordNgrams : max length of word ngram [1]
    # verbose: verbosity level [2]
    # minCount: minimal number of word occurrences [1]
    def __init__(self, lr=0.1, wordNgrams=2, verbose=2, minCount=1, epoch=25, loss='hs'):
        self.lr = lr
        self.wordNgrams = wordNgrams
        self.verbose = verbose
        self.minCount = minCount
        self.epoch = epoch
        self.loss = loss
        self.model = None

    @staticmethod
    def data_preprocess(train_df, filename='../corpus/tianchi_news/train.csv'):
        train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
        train_df[['text', 'label_ft']].to_csv(filename, index=None, header=None, sep='\t')

    def fit(self, filename='../corpus/tianchi_news/train.csv', classify=True):
        if classify:
            self.model = fasttext.train_supervised(filename, lr=self.lr, wordNgrams=self.wordNgrams, verbose=self.verbose,
                                              minCount=self.minCount, epoch=self.epoch, loss=self.loss)
        else:
            self.model = None

    def predict(self, x):
        if self.model:
            return self.model.predict(x)
        else:
            print('Please training model first!')


def test_fasttext_model():
    filename = '../corpus/tianchi_news/train.csv'
    train_df, test_df = NewsDataset().load_data()
    ft_model = FasttextModel()
    if not os.path.exists(filename):
        ft_model.data_preprocess(train_df, filename)

    ft_model.fit()
    preds = [ft_model.predict(text)[0][0].split("__")[-1] for text in test_df['text']]
    df = pd.DataFrame()
    df['label'] = preds
    df.to_csv('submit_{}.csv'.format('fasttext'), index=None)


if __name__ == '__main__':
    test_fasttext_model()










