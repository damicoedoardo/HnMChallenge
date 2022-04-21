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
from hnmchallenge.datasets.all_items_last_month_last_day import AILMLDDataset
from hnmchallenge.datasets.all_items_last_month_last_week import AILMLWDataset
from hnmchallenge.datasets.last2month_last_day import L2MLDDataset
from hnmchallenge.datasets.last_month_last_2nd_week_dataset import LML2WDataset
from hnmchallenge.datasets.last_month_last_3rd_week_dataset import LML3WDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_month_last_week_user import LMLUWDataset
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

TRAIN_PERC = 0.899
VAL_PERC = 0.1
TEST_PERC = 0.001

# VERSION = 0
# DATASET = f"dataset_v00_{VERSION}.feather"
# MODEL_NAME = f"xgb_{DATASET}.json"

# NAME = f"dataset_v1000"
NAME = f"cutf_300_ItemKNN_tw_True_rs_False"
# NAME = "cutf_100_TimePop_alpha_1.0"

VERSION = 0
# NAME = "cutf_200_TimePop_alpha_1.0"
DATASET = f"{NAME}_{VERSION}.feather"
MODEL_NAME = f"lgbm_{DATASET}.pkl"
cat = [
    "index_code_gbm",
    "product_group_name_gbm",
    "index_group_name_gbm",
    "graphical_appearance_no_gbm",
]


if __name__ == "__main__":
    save_dataset = AILMLDDataset()
    # dataset_list = [save_dataset, LML2WDataset(), LML3WDataset()]
    dataset_list = [save_dataset]

    dr = DataReader()
    # save the model on the path of the last week dataset !
    model_save_path = save_dataset._DATASET_PATH / "lgbm_models"
    model_save_path.mkdir(parents=True, exist_ok=True)

    # load into a list the datasets of the different weeks
    features_df_list = []
    for dataset in dataset_list:
        base_load_path = dataset._DATASET_PATH / "dataset_dfs/train"
        dataset_path = base_load_path / DATASET
        features_df = pd.read_feather(dataset_path)
        features_df_list.append(features_df)

    # the final features_df is the concat of the week datasets
    features_df = pd.concat(features_df_list, axis=0)
    print(features_df.columns)

    # print(features_df["ItemKNN_tw_True_rs_False_score"])
    # features_df = features_df.drop(["ItemKNN_tw_True_rs_False_rank"], axis=1)

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
        boosting_type="dart",
        objective="lambdarank",
        num_threads=72,
        # device="gpu",
        random_state=RANDOM_SEED,
        learning_rate=0.1,
        # colsample_bytree=0.8,
        reg_lambda=0.00,
        reg_alpha=0.00,
        # eta=0.05,
        num_leaves=31,
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
