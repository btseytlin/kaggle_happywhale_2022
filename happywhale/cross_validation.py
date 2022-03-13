from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold, train_test_split


class CVSplit(NamedTuple):
    train: NDArray
    val: NDArray
    test: NDArray


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
                train=train_index,
                val=val_index,
                test=test_index,
            )
        )

    return splits
