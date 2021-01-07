import time
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


class NewsVocab():
    """
    构建字典:
    train_data.keys(): ['text', 'label']
    train_data['text']: iterable of iterables
    train_data['label']: iterable
    """

    def __init__(self, train_data, min_count=5):
        self.min_count = min_count
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

    def build_vocab(self, data):
        word_counter = Counter()

        for words in data['text']:
            words = words.strip().split()
            for word in words:
                word_counter[word] += 1

        for word, count in word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        label_counter = Counter(data['label'])

        for label in range(len(label_counter)):
            count = label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def label_size(self):
        return len(self._id2label)


def preprocess(df, vocab):
    texts = df['text'].to_list()
    labels = df['label'].to_list()

    texts = [vocab.word2id(text.strip().split()) for text in texts]
    labels = vocab.label2id(labels)
    return pd.DataFrame({'text': texts, 'label': labels})


train_file = './corpus/tianchi_news/train_set.csv'
df_all = pd.read_csv(train_file, sep='\t')
# 构建词典
vocab = NewsVocab(df_all)

pre_data = preprocess(df_all, vocab)

start_time = time.time()
df_train, df_val = train_test_split(pre_data, test_size=0.2, random_state=10)
end_time = time.time()
print(f'Total time is {end_time - start_time}')



start_time = time.time()
df_train, df_val = train_test_split(df_all, test_size=0.2, random_state=10)
end_time = time.time()
print(f'Total time is {end_time - start_time}')


