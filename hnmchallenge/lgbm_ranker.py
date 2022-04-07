import math
import re

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.datasets.last_month_last_4day import LML4DDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.feature_manager import FeatureManager
from hnmchallenge.features.light_gbm_features import (
    graphical_appearance_no_gbm,
    index_code_gbm,
    index_group_name_gbm,
    product_group_name_gbm,
)
from hnmchallenge.models.itemknn.itemknn import ItemKNN

TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1

# VERSION = 0
# DATASET = f"dataset_v00_{VERSION}.feather"
# MODEL_NAME = f"xgb_{DATASET}.json"

VERSION = 0
# NAME = "dataset_letsgo4"
NAME = "cutf_200_ItemKNN_tw_True_rs_False"
DATASET = f"{NAME}_{VERSION}.feather"
MODEL_NAME = f"lgbm_{DATASET}.pkl"
cat = [
    "index_code_gbm",
    "product_group_name_gbm",
    "index_group_name_gbm",
    "graphical_appearance_no_gbm",
]


if __name__ == "__main__":
    dataset = LML4DDataset()
    dr = DataReader()
    model_save_path = dataset._DATASET_PATH / "lgbm_models"
    model_save_path.mkdir(parents=True, exist_ok=True)

    base_load_path = dataset._DATASET_PATH / "dataset_dfs/train"
    dataset_path = base_load_path / DATASET
    features_df = pd.read_feather(dataset_path)

    cat_index = [i for i, c in enumerate(features_df.columns) if c in cat]
    print(cat_index)

    for col in cat:
        features_df[col] = pd.Categorical(features_df[col])

    features_df = features_df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    # features_df = features_df.rename(columns=lambda x: re.sub("/", "", x))
    # features_df = features_df.rename(columns=lambda x: re.sub(" ", "", x))
    print(features_df.columns)

    unique_users = features_df[DEFAULT_USER_COL].unique()
    print(f"Unique users:{len(unique_users)}")
    train_len = math.ceil(len(unique_users) * TRAIN_PERC)
    val_len = math.ceil(len(unique_users) * VAL_PERC)
    test_len = math.ceil(len(unique_users) * TEST_PERC)

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_users)
    train_users, val_users, test_users = (
        unique_users[:train_len],
        unique_users[train_len : train_len + val_len],
        unique_users[train_len + val_len :],
    )

    train_df = features_df[features_df[DEFAULT_USER_COL].isin(train_users)]
    val_df = features_df[features_df[DEFAULT_USER_COL].isin(val_users)]
    test_df = features_df[features_df[DEFAULT_USER_COL].isin(test_users)]

    #####
    X_train = train_df.loc[
        :, ~train_df.columns.isin([DEFAULT_USER_COL, DEFAULT_ITEM_COL, "relevance"])
    ]
    Y_train = train_df["relevance"].copy().values
    qid_train = train_df[DEFAULT_USER_COL].copy().values

    X_val = val_df.loc[
        :, ~val_df.columns.isin([DEFAULT_USER_COL, DEFAULT_ITEM_COL, "relevance"])
    ]
    Y_val = val_df["relevance"].copy().values
    qid_val = val_df[DEFAULT_USER_COL].copy().values

    X_test = test_df.loc[
        :, ~test_df.columns.isin([DEFAULT_USER_COL, DEFAULT_ITEM_COL, "relevance"])
    ]
    Y_test = test_df["relevance"].copy().values
    qid_test = test_df[DEFAULT_USER_COL].copy().values

    query_train = train_df.groupby(DEFAULT_USER_COL)[DEFAULT_USER_COL].count()
    query_val = val_df.groupby(DEFAULT_USER_COL)[DEFAULT_USER_COL].count()

    gbm = lgb.LGBMRanker(
        boosting_type="gbdt",
        objective="lambdarank",
        # device="gpu",
        random_state=RANDOM_SEED,
        learning_rate=0.2,
        colsample_bytree=0.6,
        reg_lambda=0.0,
        reg_alpha=0.0,
        # eta=0.3,
        max_depth=8,
        n_estimators=500,
        subsample=0.8,
        # sampling_method="gradient_based"
        # n_gpus=-1
        # gpu_id=1,
    )

    gbm.fit(
        X_train,
        Y_train,
        group=query_train,
        eval_set=[(X_val, Y_val)],
        eval_group=[query_val],
        eval_metric="map",
        eval_at=12,
        verbose=True,
        early_stopping_rounds=20,
        categorical_feature=cat_index,
    )

    model_name = model_save_path / MODEL_NAME
    # save model
    joblib.dump(gbm, model_name)
    # load model
    # gbm.save_model(model_name)
