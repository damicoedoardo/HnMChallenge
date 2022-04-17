import math
import re
import time

import joblib
import lightgbm as lgb
import pandas as pd

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

SUB_NAME = "lol"

VERSION = 0
NAME = f"cutf_200_ItemKNN_tw_True_rs_False"
# NAME = "cutf_200_TimePop_alpha_1.0"
DATASET = f"{NAME}_{VERSION}.feather"
MODEL_NAME = f"lgbm_{DATASET}.pkl"
if __name__ == "__main__":
    dataset = AILMLWDataset()
    base_load_path = dataset._DATASET_PATH / "lgbm_models"
    model = joblib.load(base_load_path / MODEL_NAME)

    print("Read Dataset...")
    features = pd.read_feather(dataset._DATASET_PATH / f"dataset_dfs/full/{DATASET}")

    features = features.drop(["ItemKNN_tw_True_rs_False_rank"], axis=1)

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
    customer_article_df = features[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].copy()
    X = features.drop([DEFAULT_USER_COL, DEFAULT_ITEM_COL], axis=1)

    s = time.time()
    print("Computing Predictions...")
    y_pred = model.predict(X, num_iteration=model.best_iteration_, n_jobs=72)
    print(f"Took: {math.ceil((time.time()-s)/60)} minutes")

    customer_article_df["predicted_score"] = y_pred
    print("Sorting scores...")
    sorted_scores = customer_article_df.sort_values(
        [DEFAULT_USER_COL, "predicted_score"], ascending=[True, False]
    )
    print(sorted_scores.head(20))
    sorted_scores_index = sorted_scores.reset_index(drop=True)

    print("Filtering predictions...")
    cutoff = sorted_scores_index.groupby(DEFAULT_USER_COL).size().values
    i = 0
    filter_indices = []
    for cut in cutoff:
        filter_indices.extend(range(i, i + 12))
        i = i + cut
    final_df = sorted_scores_index.loc[filter_indices]
    final_final_df = final_df.drop("predicted_score", axis=1)

    print("Creating submission...")
    dataset.create_submission(final_final_df, sub_name=SUB_NAME)
    print("(⊙﹏⊙) Submission created succesfully (⊙﹏⊙)")
