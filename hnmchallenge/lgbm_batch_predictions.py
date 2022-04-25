import math
import re
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.datasets.all_items_last_month_last_day import AILMLDDataset
from hnmchallenge.datasets.all_items_last_month_last_week import AILMLWDataset
from hnmchallenge.datasets.last2month_last_day import L2MLDDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_month_last_week_user import LMLUWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.feature_manager import FeatureManager
from hnmchallenge.models.itemknn.itemknn import ItemKNN

SUB_NAME = "sing_day"

VERSION = 0
# NAME = f"dataset_v1000"
# NAME = "cutf_200_TimePop_alpha_1.0"
NAME = f"cutf_300_ItemKNN_tw_True_rs_False"
# NAME = "cutf_100_TimePop_alpha_1.0"
DATASET = f"{NAME}_{VERSION}.feather"
MODEL_NAME = f"lgbm_{DATASET}.pkl"
if __name__ == "__main__":
    dataset = AILMLWDataset()
    base_load_path = dataset._DATASET_PATH / "lgbm_models"
    model = joblib.load(base_load_path / MODEL_NAME)

    print("Read Dataset...")
    features = pd.read_feather(dataset._DATASET_PATH / f"dataset_dfs/full/{DATASET}")
    print(features.shape)
    # features = features.drop(["ItemKNN_tw_True_rs_False_rank"], axis=1)

    features = features.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    cat = [
        "index_code_gbm",
        "product_group_name_gbm",
        "index_group_name_gbm",
        "graphical_appearance_no_gbm",
    ]
    cat_index = [i for i, c in enumerate(features.columns) if c in cat]
    print("Categorical conversion...")
    for col in cat:
        features[col] = pd.Categorical(features[col])
    customer_article_df = features[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
    X = features.drop([DEFAULT_USER_COL, DEFAULT_ITEM_COL], axis=1)

    s = time.time()
    print("Computing Predictions...")
    y_pred = []

    batch_size = 30_000_000
    idx = 0
    while idx + batch_size < X.shape[0]:
        end = idx + batch_size
        batch = X.loc[idx : end - 1, :]
        score = model.predict(batch, num_iteration=model.best_iteration_, n_jobs=72)
        y_pred.extend(score)
        idx = end
    last_batch = X.loc[idx:, :]
    score = model.predict(last_batch, num_iteration=model.best_iteration_, n_jobs=72)
    y_pred.extend(score)
    y_pred = np.array(y_pred)
    print(y_pred.shape)

    print(f"Took: {math.ceil((time.time()-s)/60)} minutes")

    customer_article_df["predicted_score"] = y_pred
    del X
    del y_pred
    print("Sorting scores...")
    customer_article_df = customer_article_df.sort_values(
        [DEFAULT_USER_COL, "predicted_score"], ascending=[True, False]
    )
    print(customer_article_df.head(20))
    customer_article_df = customer_article_df.reset_index(drop=True)

    print("Filtering predictions...")
    cutoff = customer_article_df.groupby(DEFAULT_USER_COL).size().values
    i = 0
    filter_indices = []
    for cut in cutoff:
        filter_indices.extend(range(i, i + 12))
        i = i + cut
    customer_article_df = customer_article_df.loc[filter_indices]
    customer_article_df = customer_article_df.drop("predicted_score", axis=1)

    print("Creating submission...")
    dataset.create_submission(customer_article_df, sub_name=SUB_NAME)
    print("(⊙﹏⊙) Submission created succesfully (⊙﹏⊙)")
