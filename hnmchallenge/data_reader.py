#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os

load_dotenv()


class DataReader:
    """Utility class used to read the data"""

    _BASE_PATH = Path(os.environ.get("DATA_PATH"))
    _ARTICLES = "articles.csv"
    _CUSTOMERS = "customers.csv"
    _TRANSACTIONS = "transactions_train.csv"

    def __init__(self):
        pass

    def get_articles(self):
        path = self._BASE_PATH / self._ARTICLES
        df = pd.read_csv(path)
        return df

    def get_customer(self):
        path = self._BASE_PATH / self._CUSTOMERS
        df = pd.read_csv(path)
        return df

    def get_transactions(self):
        path = self._BASE_PATH / self._TRANSACTIONS
        df = pd.read_csv(path)
        return df
