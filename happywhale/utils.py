import os
import torch
from torch import nn
import pandas as pd
import requests
from PIL import Image
from crowdkit.aggregation import MajorityVote
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
import random
import numpy as np


def get_mlp(input_dim, output_dim, dropout_rate):
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(input_dim, input_dim),
        nn.Tanh(),
        nn.Linear(input_dim, output_dim),
    )


def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def split_tsv(df,
             val_size=0.1,
             test_size=0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, test_idx = next(splitter.split(df, df['label'], df['task']))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    test_labels = MajorityVote().fit_predict(test_df)
    test_df = pd.DataFrame({'task': test_labels.index, 'label': test_labels.values})

    assert set(train_df.task).intersection(test_df.task) == set()

    if val_size:
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size)
        train_idx, val_idx = next(splitter_val.split(train_df, train_df['label'], train_df['task']))

        val_df = train_df.iloc[val_idx]
        train_df = train_df.iloc[train_idx]
        val_labels = MajorityVote().fit_predict(val_df)
        val_df = pd.DataFrame({'task': val_labels.index, 'label': val_labels.values})

        assert set(train_df.task).intersection(val_df.task) == set()
        assert set(val_df.task).intersection(test_df.task) == set()

        return train_df, val_df, test_df

    return train_df, test_df


def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('# SEEDING DONE')
