import pandas as pd


def pd_log_show(path='./state_dict/df_log.pickle'):
    log = pd.read_pickle(path)
    return log

