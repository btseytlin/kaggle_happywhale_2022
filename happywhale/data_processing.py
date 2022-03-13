import pandas as pd
import os


def fix_train(df):
    df.species.replace({"globis": "short_finned_pilot_whale",
                        "pilot_whale": "short_finned_pilot_whale",
                        "kiler_whale": "killer_whale",
                        "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)
    return df


def load_train_test_dfs(train_images_path, test_images_path, train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path)
    train_df['image_path'] = train_df['image'].apply(lambda x: os.path.join(train_images_path, x))
    train_df['split'] = 'Train'

    train_df.species.replace({"globis": "short_finned_pilot_whale",
                        "pilot_whale": "short_finned_pilot_whale",
                        "kiler_whale": "killer_whale",
                        "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)

    test_df = pd.read_csv(test_csv_path)
    test_df['image_path'] = test_df['image'].apply(lambda x: os.path.join(test_images_path, x))
    test_df['split'] = 'Test'
    return train_df, test_df
