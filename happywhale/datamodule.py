import os
import shutil
import pandas as pd
import numpy as np
import torch
from PIL import Image
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

default_train_transforms = transforms.Compose(
    [
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

default_test_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        super(ImageDataset, self).__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.load_image(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label


class ImageDataMoodule(pl.LightningDataModule):
    def __init__(
        self,
        train_df=None,
        val_df=None,
        test_df=None,
        train_transforms=None,
        test_transforms=None,
        batch_size=32,
        num_workers=4,
        sampler=None,
        **kwargs,
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        all_individual_ids = pd.concat([train_df, val_df, test_df]).individual_id.values
        self.label_encoder = LabelEncoder().fit(all_individual_ids)

        self.train_transforms = train_transforms or default_train_transforms
        self.test_transforms = test_transforms or default_test_transforms

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler = sampler
        self.sampler_m_per_class = 4

        self.train = None
        self.val = None
        self.test = None

    def dataset_from_df(self, df, **kwargs):
        labels = self.label_encoder.transform(df.individual_id.values)
        return ImageDataset(img_paths=df.image_path.values,
                            labels=labels,
                            **kwargs)

    def setup(self, stage=None):
        if self.train_df is not None:
            self.train = self.dataset_from_df(
                self.train_df,
                transform=self.train_transforms,
            )
        if self.val_df is not None:
            self.val = self.dataset_from_df(
                self.val_df,
                transform=self.test_transforms,
            )
        if self.test_df is not None:
            self.test = self.dataset_from_df(
                self.test_df,
                transform=self.test_transforms,
            )

    def _loader_m_per_class(self, dataset):
        sampler = MPerClassSampler(
            labels=dataset.labels,
            m=self.sampler_m_per_class,
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler)

    def train_dataloader(self, sampler=None, shuffle=True):
        sampler = self.sampler if sampler is None else sampler
        if sampler == 'm_per_class':
            return self._loader_m_per_class(self.train)

        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
