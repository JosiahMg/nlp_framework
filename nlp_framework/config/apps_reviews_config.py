# -*- coding: utf-8 -*-
# file: apps_reviews_config.py
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


EPOCHS = 10
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
RANDOM_SEED = 42
BATCH_SIZE = 2
MAX_LEN = 160
LR = 2e-5
PATH = 'data/apps_reviews.csv'
MODEL_NAME = 'apps_reviews_state.bin'
class_names = ['negative', 'neutral', 'positive']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
