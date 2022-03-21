import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.feature_manager import FeatureManager
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.ease.ease import EASE
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models.sgmc.sgmc import SGMC
from hnmchallenge.models.top_pop import TopPop
from hnmchallenge.stratified_dataset import StratifiedDataset

TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1
DATASET = "dataset_v8.feather"

MODEL_NAME = "xgb_v8.json"

if __name__ == "__main__":
    dataset = StratifiedDataset()
    dr = DataReader()
    model_save_path = dr.get_preprocessed_data_path() / "xgb_models"
    model_save_path.mkdir(parents=True, exist_ok=True)

    base_load_path = dr.get_preprocessed_data_path() / "xgb_datasets"
    dataset_path = base_load_path / DATASET
    features_df = pd.read_feather(dataset_path)

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

    model = xgb.XGBRanker(
        tree_method="gpu_hist",
        booster="gbtree",
        objective="rank:map",
        random_state=RANDOM_SEED,
        learning_rate=0.2,
        colsample_bytree=0.8,
        reg_lambda=0.0,
        reg_alpha=0.0,
        eta=0.1,
        max_depth=5,
        n_estimators=500,
        subsample=0.8,
        # sampling_method="gradient_based"
        # n_gpus=-1
        # gpu_id=1,
    )

    model.fit(
        X_train,
        Y_train,
        qid=qid_train,
        eval_set=[(X_val, Y_val)],
        eval_qid=[qid_val],
        eval_metric=["map@12"],
        verbose=True,
        early_stopping_rounds=15,
    )

    model_name = model_save_path / MODEL_NAME
    model.save_model(model_name)
