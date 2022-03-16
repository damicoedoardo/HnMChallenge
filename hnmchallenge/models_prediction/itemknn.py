import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models.top_pop import TopPop
from hnmchallenge.stratified_dataset import StratifiedDataset

KIND = "full"
CUTOFF = 40
TIME_WEIGHT = True
assert KIND in ["train", "full"], "kind should be train or full"

if __name__ == "__main__":
    dataset = StratifiedDataset()
    dr = DataReader()

    recom = ItemKNN(dataset, topk=1000, time_weight=True)

    holdin = dataset.get_holdin()
    fd = dr.get_filtered_full_data()

    # set the correct data to use
    data_df = None
    if KIND == "train":
        data_df = holdin
    else:
        data_df = fd

    recom.compute_similarity_matrix(data_df)
    recs = recom.recommend_multicore(
        interactions=data_df,
        batch_size=10_000,
        num_cpus=20,
        remove_seen=False,
        white_list_mb_item=None,
        cutoff=CUTOFF,
    )

    # recs_list = recs.groupby(DEFAULT_USER_COL)["article_id"].apply(list).to_frame()
    recs = recs.rename(
        {"article_id": "recs", "prediction": "itemknn_score", "rank": "itemknn_rank"},
        axis=1,
    )
    recs.reset_index(drop=True).to_feather(
        dr.get_preprocessed_data_path()
        / f"{KIND}_cosine_recs_{CUTOFF}_tw_{TIME_WEIGHT}.feather"
    )
