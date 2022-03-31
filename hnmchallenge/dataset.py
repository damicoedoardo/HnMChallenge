from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.utils.sparse_matrix import interactions_to_sparse_matrix


class Dataset:
    _ARTICLES_NUM = 104547  # total number of items in the full data
    _CUSTOMERS_NUM = 1362281  # total number of users in the full data

    def __init__(self, kind: str = "STANDARD") -> None:
        dataset_kinds = ["STANDARD", "SMALL"]
        assert kind in dataset_kinds, f"dataset kind must be in {dataset_kinds}"
        self.kind = kind
        self.train, self.val, self.test = self._load_splits()
        (
            self.train_user_subset,
            self.val_user_subset,
            self.test_user_subset,
        ) = self._load_splits_user_subset()

    def _load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train validation and test data as dataframes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val and test df
        """
        dr = DataReader()
        pdp = dr.get_preprocessed_data_path()

        train_name = "train_df.feather" if self.kind == "STANDARD" else "small_train_df"
        val_name = "val_df.feather"
        test_name = "test_df.feather"

        data_df_list = []
        for data_split_name in [train_name, val_name, test_name]:
            data_df_list.append(pd.read_feather(pdp / data_split_name))

        train, val, test = data_df_list
        return train, val, test

    def _load_splits_user_subset(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train validation and test data as dataframes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val and test df
        """
        dr = DataReader()
        pdp = dr.get_preprocessed_data_path()

        train_name = "train_df_user_subset.feather"
        val_name = "val_df_user_subset.feather"
        test_name = "test_df_user_subset.feather"

        data_df_list = []
        for data_split_name in [train_name, val_name, test_name]:
            data_df_list.append(pd.read_feather(pdp / data_split_name))

        train, val, test = data_df_list
        return train, val, test

    def get_train_df(self) -> pd.DataFrame:
        return self.train

    def get_val_df(self) -> pd.DataFrame:
        return self.val

    def get_test_df(self) -> pd.DataFrame:
        return self.test

    def get_train_df_user_subset(self) -> pd.DataFrame:
        return self.train_user_subset

    def get_val_df_user_subset(self) -> pd.DataFrame:
        return self.val_user_subset

    def get_test_df_user_subset(self) -> pd.DataFrame:
        return self.test_user_subset

    def get_user_item_interaction_matrix(
        self, interaction_df: pd.DataFrame
    ) -> sps.coo_matrix:
        """Create the user-item interaction matrix for the given interaction df

        Args:
            interaction_df (pd.DataFrame): interaction df from which build the user-item interaction matrix

        Returns:
            sps.coo_matrix: user-item interaction matrix in coo format
        """
        sp_m, _, _ = interactions_to_sparse_matrix(
            interactions=interaction_df,
            users_num=self._CUSTOMERS_NUM,
            items_num=self._ARTICLES_NUM,
        )
        return sp_m

    def get_black_list_item(self) -> np.array:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        with open(p / "item_last_month.npy", "rb") as f:
            item_last_month = np.load(f)
        ar = np.arange(self._ARTICLES_NUM)
        black_list_item = ar[~np.isin(ar, item_last_month)]
        return black_list_item

    def get_holdout(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_df = pd.read_feather(
            p / "full_holdout_last_week.feather",
        )
        return holdout_df

    def get_holdin(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_df = pd.read_feather(
            p / "full_holdin_last_week.feather",
        )
        return holdout_df

    def save_holdout_groundtruth(self) -> None:
        """Save the dataframe containing the groundtruth for every user in the holdout set (last week)"""
        dr = DataReader()
        p = dr.get_preprocessed_data_path()

        # Add relevance column on dataframe
        # retrieve the holdout
        holdout = self.get_holdout()
        # retrieve items per user in holdout
        item_per_user = holdout.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
        item_per_user_df = item_per_user.to_frame()
        # items groundtruth
        items_groundtruth = (
            item_per_user_df.reset_index().explode(DEFAULT_ITEM_COL).drop_duplicates()
        )
        items_groundtruth.reset_index(drop=True).to_feather(
            p / "full_holdout_groundtruth.feather"
        )

    def get_holdout_groundtruth(self) -> pd.DataFrame:
        dr = DataReader()
        p = dr.get_preprocessed_data_path()
        holdout_gt = pd.read_feather(p / "full_holdout_groundtruth.feather")
        return holdout_gt


if __name__ == "__main__":
    d = Dataset()
    d.save_holdout_groundtruth()
    holdout = d.get_holdout_groundtruth()
    print(holdout[DEFAULT_USER_COL].nunique())
