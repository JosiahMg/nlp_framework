import logging
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from transformers import *
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

PATH = 'pre_trained_models/chinese-bert-base-tf/'
BERT_PATH = 'pre_trained_models/chinese-bert-base-tf/'
WEIGHT_PATH = 'pre_trained_models/chinese-bert-base-tf/'
MAX_SEQUENCE_LENGTH = 128
input_categories = ['q1','q2']
output_categories = 'label'


train_left = pd.read_csv('corpus/estate/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('corpus/estate/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('好的')
# for test
df_train = df_train[:100]

test_left = pd.read_csv('corpus/estate/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right =  pd.read_csv('corpus/estate/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
df_test = test_left.merge(test_right, how='left')
# for test
df_test = df_test[:100]

print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       truncation=True
                                       )

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.q1, instance.q2

        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30, 60):
        tres = i / 100
        y_pred_bin = (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t

tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-chinese-vocab.txt')
outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = BertConfig.from_pretrained(BERT_PATH+'bert-base-chinese-config.json')
    config.output_hidden_states = False
    bert_model = TFBertModel.from_pretrained(BERT_PATH+'bert-base-chinese-tf_model.h5', config=config)
    all_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)
    q_embedding = all_embedding[0]
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalMaxPooling1D()(q_embedding)
    t = q_embedding[:, -1]
    e = q_embedding[:, 0]
    x = tf.keras.layers.Concatenate()([q, a, t, e])

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn], outputs=x)

    return model


gkf = GroupKFold(n_splits=5).split(X=df_train.q2, groups=df_train.id)
valid_preds = []
test_preds = []
oof = np.zeros((len(df_train), 1))

for fold, (train_idx, valid_idx) in enumerate(gkf):
  train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
  train_outputs = outputs[train_idx]
  valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
  valid_outputs = outputs[valid_idx]

  K.clear_session()
  model = create_model()
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
  model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=[tf.keras.metrics.AUC()])
  model.fit(train_inputs, train_outputs, validation_data = (valid_inputs, valid_outputs), epochs=1, batch_size=2)
  oof_p = model.predict(valid_inputs, batch_size=512)
  oof[valid_idx] = oof_p
  valid_preds.append(oof_p)
  test_preds.append(model.predict(test_inputs, batch_size=512))
  f1,t = search_f1(valid_outputs, valid_preds[-1])
  print('validation score = ', f1)

best_score, best_t = search_f1(outputs, oof)

sub = np.average(test_preds, axis=0)
sub = sub > best_t
df_test['label'] = sub.astype(int)
df_test[['id','id_sub','label']].to_csv('submission_beike_{}.csv'.format(best_score),index=False, header=None,sep='\t')
