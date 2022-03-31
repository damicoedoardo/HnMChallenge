from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.utils.sparse_matrix import interactions_to_sparse_matrix


class StratifiedDataset:
    _ARTICLES_NUM = 22_069  # total number of items in the full data
    _CUSTOMERS_NUM = 1_136_206  # total number of users in the full data

    def __init__(self, percentages_splits: str = "80_20") -> None:
        self.percentages_splits = percentages_splits

    def get_holdin(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdin_df = pd.read_feather(
            p / f"stratified_holdin_{self.percentages_splits}.feather",
        )
        return holdin_df

    def get_holdout(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_df = pd.read_feather(
            p / f"stratified_holdout_{self.percentages_splits}.feather",
        )
        return holdout_df

    def get_last_month_holdin(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdin_df = pd.read_feather(
            p / f"last_month_holdin.feather",
        )
        return holdin_df

    def get_last_month_holdout(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_df = pd.read_feather(
            p / f"last_month_holdout.feather",
        )
        return holdout_df

    def get_last_day_holdin(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdin_df = pd.read_feather(
            p / f"last_day_holdin.feather",
        )
        return holdin_df

    def get_last_day_holdout(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_df = pd.read_feather(
            p / f"last_day_holdout.feather",
        )
        return holdout_df

    def save_holdout_groundtruth(self) -> None:
        """Save the dataframe containing the groundtruth for every user in the holdout set (last week)"""
        dr = DataReader()
        p = dr.get_preprocessed_data_path()

        # Add relevance column on dataframe
        # retrieve the holdout
        holdout = self.get_last_day_holdout()
        # retrieve items per user in holdout
        item_per_user = holdout.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
        item_per_user_df = item_per_user.to_frame()
        # items groundtruth
        items_groundtruth = (
            item_per_user_df.reset_index().explode(DEFAULT_ITEM_COL).drop_duplicates()
        )
        items_groundtruth.reset_index(drop=True).to_feather(
            p / "holdout_groundtruth.feather"
        )

    def get_holdout_groundtruth(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_gt = pd.read_feather(p / "holdout_groundtruth.feather")
        return holdout_gt


if __name__ == "__main__":
    d = StratifiedDataset()
    d.save_holdout_groundtruth()
    holdout = d.get_holdout_groundtruth()
    print(holdout[DEFAULT_USER_COL].nunique())
