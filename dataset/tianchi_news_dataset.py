import numpy as np
import pandas as pd

"""
for model of count, tifid and fasttext
"""
class NewsDataset:
    def __init__(self):
        self.train_df = pd.read_csv('../corpus/tianchi_news/train_set.csv', sep='\t')
        self.test_df = pd.read_csv('../corpus/tianchi_news/test_a.csv', sep='\t')

    def load_data(self):
        return self.train_df, self.test_df


"""
for model of textcnn
"""
class NewsSplitDataset:
    def __init__(self):
        self.df_all = pd.read_csv('../corpus/tianchi_news/train_set.csv', sep='\t', nrows=100)
        # self.test_df = pd.read_csv('../corpus/tianchi_news/test_a.csv', sep='\t')





