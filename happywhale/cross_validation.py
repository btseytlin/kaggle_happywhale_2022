
from typing import NamedTuple, List
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


class CVSplit(NamedTuple):
    train: List
    val: List
    test: List


def get_cv_splits(train_df,
                  val_size=0.1,
                  n_folds=5):
    index = range(len(train_df))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

    splits = []
    for train_index, test_index in cv.split(index, train_df.individual_id.values):
        train_index, val_index = train_test_split(train_index,
                                                  test_size=val_size)

        splits.append(
            CVSplit(
                train=list(train_index),
                val=list(val_index),
                test=list(test_index),
            )
        )

    return splits
