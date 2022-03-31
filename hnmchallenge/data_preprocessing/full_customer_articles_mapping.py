import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.evaluation.python_evaluation import map_at_k
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.top_pop import TopPop

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset = Dataset()
    dr = DataReader()

    raw_articles = dr.get_articles()
    raw_customers = dr.get_customer()

    print(raw_articles[DEFAULT_ITEM_COL].nunique())
    print(raw_customers[DEFAULT_USER_COL].nunique())

    # map the user and item columns raw ids
    user_mapping_dict, item_mapping_dict = dr.get_raw_new_mapping_dict()
    raw_articles[DEFAULT_ITEM_COL] = raw_articles[DEFAULT_ITEM_COL].map(
        item_mapping_dict
    )
    raw_customers[DEFAULT_USER_COL] = raw_customers[DEFAULT_USER_COL].map(
        user_mapping_dict
    )

    print(raw_articles[DEFAULT_ITEM_COL].nunique())
    print(raw_customers[DEFAULT_USER_COL].nunique())

    raw_articles.reset_index(drop=True).to_feather(
        dr.get_preprocessed_data_path() / "full_articles.feather"
    )
    raw_customers.reset_index(drop=True).to_feather(
        dr.get_preprocessed_data_path() / "full_customers.feather"
    )
