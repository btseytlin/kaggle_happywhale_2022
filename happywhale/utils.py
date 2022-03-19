import os
import torch
from torch import nn
import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm
import random
import numpy as np


def get_mlp(input_dim, output_dim, dropout_rate, normalize=True):
    layers = [
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
    ]
    if normalize:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch
