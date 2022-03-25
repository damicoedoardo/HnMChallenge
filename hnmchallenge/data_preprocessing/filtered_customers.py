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
    dataset = FilterdDataset()
    dr = DataReader()

    full_customers = dr.get_customer()
    full_customers = full_customers.fillna(0)
    filtered = dr.get_filtered_new_raw_mapping_dict()
    cust_dict, article_dict = filtered

    customer_frame = pd.DataFrame.from_dict(cust_dict, orient="index")
    customer_frame = customer_frame.reset_index(level=0)
    customer_frame = customer_frame.rename(columns={0: "customer_id"})
    customer = customer_frame.merge(full_customers, on="customer_id", how="left")
    customer = customer.drop("customer_id", axis=1)
    customer = customer.rename(columns={"index": "customer_id"})

    customer["club_member_status"] = customer["club_member_status"].astype(str)
    customer["fashion_news_frequency"] = customer["fashion_news_frequency"].astype(str)

    customer.to_feather(dr.get_preprocessed_data_path() / "filtered_customers.feather")
