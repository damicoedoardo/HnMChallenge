import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

from hnmchallenge.data_reader import DataReader

# load env variables
load_dotenv()


class DatasetInterface(ABC):
    _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))
    _SUBMISSION_FOLDER = Path(__file__).parent.parent / "submission"

    DATASET_NAME = None
    _DATASET_DESCRIPTION = None

    _DATASET_PATH = None
    _MAPPING_DICT_PATH = Path(_DATASET_PATH / "mapping_dict")

    _ARTICLE_PATH = _DATASET_PATH / "articles.feather"
    _CUSTOMER_PATH = _DATASET_PATH / "customers.feather"
    _HOLDIN_PATH = _DATASET_PATH / "holdin.feather"
    _HOLDOUT_PATH = _DATASET_PATH / "holdout.feather"
    _FULL_DATA_PATH = _DATASET_PATH / "full_data.feather"

    # this should be hardcoded
    _ARTICLES_NUM = None
    _CUSTOMERS_NUM = None

    def __init__(self) -> None:
        assert self.DATASET_NAME is not None, "Dataset name has not been set"

        # create and store the dataset description
        self._DATASET_DESCRIPTION = self.create_dataset_description()
        # create the dataset path
        self._DATASET_PATH = self._DATA_PATH / "datasets" / self.DATASET_NAME
        # create the dataset folder
        self._DATASET_PATH.mkdir(parents=True, exist_ok=True)
        self.dr = DataReader()

    def __str__(self):
        """Rerturn the description of the dataset"""
        return self._DATASET_DESCRIPTION

    def get_dataset_path(self) -> Path:
        return self._DATASET_PATH

    @abstractmethod
    def remap_user_item_ids() -> None:
        """
        Remap user item ids on transaction customer and articles df and creates the mapping dictionary
        - store full_data
        - create and save mapping dict for user and item ids
        - create customer and articles df
        - update the user and item dictionaries with the missing users and items
        """
        pass

    @abstractmethod
    def create_holdin_holdout() -> None:
        """Create holdin and holdout and save them"""
        pass

    @abstractmethod
    def create_dataset_description(self) -> str:
        """Create the dataset description"""
        pass

    def get_holdin(self) -> pd.DataFrame:
        """Return the holdin for the dataset"""
        df = pd.read_feather(self._HOLDIN_PATH)
        return df

    def get_holdout(self) -> pd.DataFrame:
        """Return the holdout for the dataset"""
        df = pd.read_feather(self._HOLDOUT_PATH)
        return df

    def get_full_data(self) -> pd.DataFrame:
        """Return the full dataset"""
        df = pd.read_feather(self._FULL_DATA_PATH)
        return df

    def get_customers_df(self) -> pd.DataFrame:
        """Return the customer df with user feature"""
        df = pd.read_feather(self._CUSTOMER_PATH)
        return df

    def get_articles_df(self) -> pd.DataFrame:
        """Return the article df with item feature"""
        df = pd.read_feather(self._ARTICLE_PATH)
        return df

    def get_raw_new_mapping_dict(self) -> Tuple[dict, dict]:
        """Return the RAW -> NEW mapping dict"""
        uids_p = self._MAPPING_DICT_PATH / "raw_new_user_ids_dict.pkl"
        iids_p = self._MAPPING_DICT_PATH / "raw_new_item_ids_dict.pkl"
        with open(uids_p, "rb") as f:
            uids_dict = pickle.load(f)
        with open(iids_p, "rb") as f:
            iids_dict = pickle.load(f)
        return uids_dict, iids_dict

    def get_new_raw_mapping_dict(self) -> Tuple[dict, dict]:
        """Return the NEW -> RAW mapping dict"""
        uids_p = self._MAPPING_DICT_PATH / "filtered_new_raw_user_ids_dict.pkl"
        iids_p = self._MAPPING_DICT_PATH / "filtered_new_raw_item_ids_dict.pkl"
        with open(uids_p, "rb") as f:
            uids_dict = pickle.load(f)
        with open(iids_p, "rb") as f:
            iids_dict = pickle.load(f)
        return uids_dict, iids_dict

    @abstractmethod
    def create_submission(self, recs_df: pd.DataFrame, sub_name: str) -> None:
        """
        Create submission given the recommendation for the users
        Check which users are missing compute for those the popularity recommendations
        and merge the two recommendations df
        """
        # TODO can be a non abstract method done directly on the dataset interface
        pass
