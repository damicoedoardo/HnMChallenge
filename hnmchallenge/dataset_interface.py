import os
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
    _DATASET_PATH = None

    _ARTICLES_NUM = None
    _CUSTOMERS_NUM = None

    def __init__(self) -> None:
        # create the dataset path
        self._DATASET_PATH = self._DATA_PATH / "datasets" / self.DATASET_NAME
        # create the dataset folder
        self._DATASET_PATH.mkdir(parents=True, exist_ok=True)
        self.dr = DataReader()

    def get_dataset_path(self) -> Path:
        return self._DATASET_PATH

    @abstractmethod
    def get_holdin(self) -> pd.DataFrame:
        """Return the holdin for the dataset"""
        pass

    @abstractmethod
    def get_holdout(self) -> pd.DataFrame:
        """Return the holdout for the dataset"""
        pass

    @abstractmethod
    def get_full_data(self) -> pd.DataFrame:
        """Return the full dataset"""
        pass

    @abstractmethod
    def get_customers_df(self) -> pd.DataFrame:
        """Return the customer df with user feature"""
        pass

    @abstractmethod
    def get_articles_df(self) -> pd.DataFrame:
        """Return the article df with item feature"""
        pass

    @abstractmethod
    def get_raw_new_mapping_dict(self) -> Tuple(dict, dict):
        """Return the RAW -> NEW mapping dict"""
        pass

    @abstractmethod
    def get_new_raw_mapping_dict(self) -> Tuple(dict, dict):
        """Return the NEW -> RAW mapping dict"""
        pass

    @abstractmethod
    def create_submission(self, recs_df: pd.DataFrame, sub_name: str) -> None:
        """
        Create submission given the recommendation for the users
        Check which users are missing and load for those the popularity recommendations
        """
        pass
