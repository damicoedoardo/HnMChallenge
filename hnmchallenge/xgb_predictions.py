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
from hnmchallenge.submission_handler import SubmissionHandler

MODEL_NAME = "xgb_v4.json"
SUBMISSION_NAME = "XGB_v4"


def plot(model):
    fig, ax = plt.subplots(figsize=(20, 20))
    plot_importance(model, ax=ax)
    plt.show()


def filtered_indices(data):
    cutoff = data.groupby(DEFAULT_USER_COL).size().values
    i = 0
    filter_indices = []
    for cut in cutoff:
        filter_indices.extend(range(i, i + 11))
        i = i + cut

    return filter_indices


if __name__ == "__main__":
    dataset = StratifiedDataset()
    dr = DataReader()
    base_load_path = dr.get_preprocessed_data_path() / "xgb_models"
    model = xgb.XGBRanker()
    model.load_model(base_load_path / MODEL_NAME)
    plot(model)
    features = pd.read_feather(
        dr.get_preprocessed_data_path()
        / "xgb_predictions_datasets"
        / "dataset_v4.feather"
    )

    customer_article_df = features[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].copy()
    X = features.drop([DEFAULT_USER_COL, DEFAULT_ITEM_COL], axis=1)
    y_pred = model.predict(X, ntree_limit=model.best_ntree_limit)
    customer_article_df["predicted_score"] = y_pred
    sorted_scores = customer_article_df.sort_values(
        [DEFAULT_USER_COL, "predicted_score"], ascending=False
    )

    sorted_scores_index = sorted_scores.reset_index(drop=True)
    indices = filtered_indices(sorted_scores_index)
    final_df = sorted_scores_index.loc[indices]
    final_final_df = final_df.drop("predicted_score", axis=1)
    sh = SubmissionHandler()
    sh.create_submission_filtered_data([final_final_df], sub_name=SUBMISSION_NAME)
