from typing import Tuple
import pandas as pd
import numpy as np
import scipy.sparse as sps

from hnmchallenge.data_reader import DataReader


class Dataset:
    _ARTICLES_NUM = 104547  # total number of items in the full data
    _CUSTOMERS_NUM = 1362281  # total number of users in the full data

    def __init__(self, kind: str = "STANDARD") -> None:
        dataset_kinds = ["STANDARD", "SMALL"]
        assert kind in dataset_kinds, f"dataset kind must be in {dataset_kinds}"
        self.kind = kind

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the train validation and test data as dataframes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val and test df
        """
        dr = DataReader()
        pdp = dr.get_preprocessed_data_path()

        train_name = "train_df" if self.kind == "STANDARD" else "small_train_df"
        val_name = "val_df"
        test_name = "test_df"

        for data_split_name in [train_name, val_name, test_name]:
            pd.read_feather(pdp / data_split_name)

    def get_user_item_interaction_matrix(
        self, interaction_df: pd.DataFrame
    ) -> sps.coo_matrix:
        """Create the user-item interaction matrix for the given interaction df

        Args:
            interaction_df (pd.DataFrame): interaction df from which build the user-item interaction matrix

        Returns:
            sps.coo_matrix: user-item interaction matrix in coo format
        """
        pass
