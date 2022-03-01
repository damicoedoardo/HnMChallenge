from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

from hnmchallenge.data_reader import DataReader
from hnmchallenge.utils.sparse_matrix import interactions_to_sparse_matrix


class FilterdDataset:
    # TODO: let's finish that.
    _ARTICLES_NUM = 22_069  # total number of items in the full data
    _CUSTOMERS_NUM = 1_136_206  # total number of users in the full data

    def __init__(self) -> None:
        self.train, self.val, self.test = self._load_splits()
        self.train_user_subset = self._load_splits_user_subset()

    def _load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train validation and test data as dataframes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val and test df
        """
        dr = DataReader()
        pdp = dr.get_preprocessed_data_path()

        train_name = "filtered_train_df.feather"
        val_name = "filtered_val_df.feather"
        test_name = "filtered_test_df.feather"

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

        train_name = "filtered_train_small_df.feather"
        train = pd.read_feather(pdp / train_name)
        return train

    def get_train_df(self) -> pd.DataFrame:
        return self.train

    def get_val_df(self) -> pd.DataFrame:
        return self.val

    def get_test_df(self) -> pd.DataFrame:
        return self.test

    def get_train_df_user_subset(self) -> pd.DataFrame:
        return self.train_user_subset

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
