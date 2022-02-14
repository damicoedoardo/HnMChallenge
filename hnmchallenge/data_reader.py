#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import logging
import os
import pickle
from operator import index
from pathlib import Path
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

from hnmchallenge.constant import *
from hnmchallenge.utils.logger import set_color

load_dotenv()
logger = logging.getLogger(__name__)


class DataReader:

    _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))
    _PREPROCESSED_DATA_PATH = Path(_DATA_PATH / "preprocessed")
    _MAPPING_DICT_PATH = Path(_PREPROCESSED_DATA_PATH / "mapping_dict")
    _ARTICLES = Path("articles.csv")
    _CUSTOMERS = Path("customers.csv")
    _TRANSACTIONS = Path("transactions_train.csv")
    _SAMPLE_SUBMISSION = Path(_DATA_PATH / "sample_submission.csv")
    _SUBMISSION_FOLDER = Path(__file__).parent.parent / "submission"

    def __init__(self):
        # create the dir for preprocessed files
        pass

    def get_data_path(self) -> Path:
        return self._DATA_PATH

    def get_preprocessed_data_path(self) -> Path:
        return self._PREPROCESSED_DATA_PATH

    def get_mapping_dict_path(self) -> Path:
        return self._MAPPING_DICT_PATH

    def get_submission_folder(self) -> Path:
        return self._SUBMISSION_FOLDER

    def ensure_dirs(self) -> None:
        """Create the necessary dirs for the preprocessed data"""
        self.get_preprocessed_data_path().mkdir(parents=True, exist_ok=True)
        self.get_mapping_dict_path().mkdir(parents=True, exist_ok=True)
        self.get_submission_folder().mkdir(parents=True, exist_ok=True)

    def get_new_raw_mapping_dict(self) -> Tuple[dict, dict]:
        uids_p = self.get_mapping_dict_path() / "new_raw_user_ids_dict.pkl"
        iids_p = self.get_mapping_dict_path() / "new_raw_item_ids_dict.pkl"
        with open(uids_p, "rb") as f:
            uids_dict = pickle.load(f)
        with open(iids_p, "rb") as f:
            iids_dict = pickle.load(f)
        return uids_dict, iids_dict

    def get_raw_new_mapping_dict(self) -> Tuple[dict, dict]:
        uids_p = self.get_mapping_dict_path() / "raw_new_user_ids_dict.pkl"
        iids_p = self.get_mapping_dict_path() / "raw_new_item_ids_dict.pkl"
        with open(uids_p, "rb") as f:
            uids_dict = pickle.load(f)
        with open(iids_p, "rb") as f:
            iids_dict = pickle.load(f)
        return uids_dict, iids_dict

    def get_sample_submission(self) -> pd.DataFrame:
        ss = pd.read_csv(self._SAMPLE_SUBMISSION)
        return ss

    def get_target_user(self) -> pd.DataFrame:
        target_user = pd.read_feather(
            self.get_preprocessed_data_path() / "target_user.feather"
        )
        return target_user

    def get_zero_interatction_users(self) -> pd.DataFrame:
        zero_interaction_users = pd.read_feather(
            self.get_preprocessed_data_path() / "zero_int_users.feather"
        )
        return zero_interaction_users

    def get_full_data(self) -> pd.DataFrame:
        p = self.get_preprocessed_data_path() / "transactions.feather"
        df = pd.read_feather(p)
        # convert date to pandas datetime
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        return df

    def get_articles(self) -> pd.DataFrame:
        path = self.get_data_path() / self._ARTICLES
        df = pd.read_csv(path, dtype={"article_id": str})
        return df

    def get_customer(self) -> pd.DataFrame:
        path = self.get_data_path() / self._CUSTOMERS
        df = pd.read_csv(path)
        return df

    def get_transactions(self) -> pd.DataFrame:
        path = self.get_data_path() / self._TRANSACTIONS
        df = pd.read_csv(path, dtype={"article_id": str})
        return df
