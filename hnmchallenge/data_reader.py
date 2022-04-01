#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import logging
import os
import pickle
from operator import index
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from hnmchallenge.constant import *
from hnmchallenge.utils.logger import set_color

load_dotenv()
logger = logging.getLogger(__name__)


class DataReader:

    _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))
    _PREPROCESSED_DATA_PATH = Path(_DATA_PATH / "preprocessed")
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

    def get_submission_folder(self) -> Path:
        return self._SUBMISSION_FOLDER

    def ensure_dirs(self) -> None:
        """Create the necessary dirs for the preprocessed data"""
        self.get_preprocessed_data_path().mkdir(parents=True, exist_ok=True)
        self.get_mapping_dict_path().mkdir(parents=True, exist_ok=True)
        self.get_submission_folder().mkdir(parents=True, exist_ok=True)

    def get_sample_submission(self) -> pd.DataFrame:
        ss = pd.read_csv(self._SAMPLE_SUBMISSION)
        return ss

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
