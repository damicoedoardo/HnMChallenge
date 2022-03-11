from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

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

    def get_features_holdin_user_item(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        useritem_df = pd.read_feather(
            p / f"stratified_holdin_useritem_dataset.feather",
        )
        return useritem_df

    def get_features_holdin_item(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        item_df = pd.read_feather(
            p / f"stratified_holdin_item_dataset.feather",
        )
        return item_df