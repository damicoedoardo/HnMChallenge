import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.ease.ease import EASE
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models.sgmc.sgmc import SGMC
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color

KIND = "train"
CUTOFF = 250

RECS_NAME = f"{KIND}_itemknn_{CUTOFF}_tw_True.feather"

if __name__ == "__main__":
    dataset = StratifiedDataset()
    dr = DataReader()

    # retrieve the prediction of the model
    recs = pd.read_feather(dr.get_preprocessed_data_path() / RECS_NAME)

    # retrieve the holdout
    holdout = dataset.get_last_day_holdout()
    # retrieve items per user in holdout
    item_per_user = holdout.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
    item_per_user_df = item_per_user.to_frame()
    # items groundtruth
    items_groundtruth = (
        item_per_user_df.reset_index().explode(DEFAULT_ITEM_COL).drop_duplicates()
    )

    # merge recs and item groundtruth
    merged = pd.merge(
        recs,
        items_groundtruth,
        left_on=[DEFAULT_USER_COL, "recs"],
        right_on=[DEFAULT_USER_COL, "article_id"],
        how="left",
    )

    # we have to remove the user for which we do not do at least one hit,
    # since we would not have the relavance for the items
    merged.loc[merged["article_id"].notnull(), "article_id"] = 1
    merged["hit_sum"] = merged.groupby(DEFAULT_USER_COL)["article_id"].transform("sum")

    merged_filtered = merged[merged["hit_sum"] > 0]

    # we can drop the hit sum column
    merged_filtered = merged_filtered.drop("hit_sum", axis=1)

    # fill with 0 the nan values, the nan are the one for which we do not do an hit
    merged_filtered["article_id"] = merged_filtered["article_id"].fillna(0)

    # rename the columns
    merged_filtered = merged_filtered.rename(
        {"recs": DEFAULT_ITEM_COL, "article_id": "relevance"}, axis=1
    ).reset_index(drop=True)

    BASE_SAVE_PATH = dr.get_preprocessed_data_path() / "relevance_dfs"
    BASE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    merged_filtered.to_feather(BASE_SAVE_PATH / RECS_NAME)
