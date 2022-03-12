import pandas as pd
from hnmchallenge.data_reader import DataReader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hnmchallenge.dataset import Dataset
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.top_pop import TopPop
from hnmchallenge.evaluation.python_evaluation import map_at_k
from hnmchallenge.constant import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dataset = FilterdDataset()
dr = DataReader()

full_customers=dr.get_customer()
full_customers=full_customers.fillna(0)
filtered=dr.get_filtered_new_raw_mapping_dict()
cust_dict, article_dict = filtered

customer_frame= pd.DataFrame.from_dict(cust_dict,orient='index')
customer_frame=customer_frame.reset_index(level=0)
customer_frame=customer_frame.rename(columns={0:"customer_id"})
customer=customer_frame.merge(full_customers,on='customer_id',how='left')
customer=customer.drop("customer_id",axis=1)
customer=customer.rename(columns={"index":"customer_id"})

customer.to_feather(
        dr.get_preprocessed_data_path() / "filtered_customers.feather"
    )