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

full_articles=dr.get_articles()
filtered=dr.get_filtered_new_raw_mapping_dict()
cust_dict, article_dict = filtered

article_frame= pd.DataFrame.from_dict(article_dict,orient='index')
article_frame=article_frame.reset_index(level=0)
article_frame=article_frame.rename(columns={0:"article_id"})

article=article_frame.merge(full_articles,on='article_id',how='left')
article=article.drop("article_id",axis=1)
article=article.rename(columns={"index":"article_id"})


article.to_feather(
        dr.get_preprocessed_data_path() / "filtered_artilces.feather"
    )