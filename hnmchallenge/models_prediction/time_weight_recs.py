import pandas as pd
from hnmchallenge.data_reader import DataReader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hnmchallenge.dataset import Dataset
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.top_pop import TopPop
from hnmchallenge.evaluation.python_evaluation import map_at_k
from hnmchallenge.constant import *
from hnmchallenge.models_prediction.recs_interface import RecsInterface

import logging


class TimeWeight(RecsInterface):
    def __init__(
        self,
        kind: str,
        dataset: StratifiedDataset,
        cutoff: int = 0,
        # alpha: float = 0.95,
        # eps: float = 1e-6,
    ) -> None:
        super().__init__(kind, dataset, cutoff)
        # self.alpha = alpha
        # self.eps = eps

        self.RECS_NAME = f"TimeWeight"

    def get_recommendations(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_last_month_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        # data that you use to compute similarity
        # Using the full data available perform better
        last_month_data = data_df[data_df["t_dat"] > "2020-08-31"]

        # drop multiple buys
        # data_df = data_df.drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL])

        df = data_df
        df["last_buy"] = df.groupby(DEFAULT_USER_COL)["t_dat"].transform(max)
        df["first_buy"] = df.groupby(DEFAULT_USER_COL)["t_dat"].transform(min)
        df["time_score"] = (df["t_dat"] - df["first_buy"]) / (
            df["last_buy"] - df["first_buy"]
        )
        df = df.fillna(1)
        df["rank_time"] = (
            df.groupby(DEFAULT_USER_COL)["time_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        df = df.drop(
            [
                "t_dat",
                "price",
                "sales_channel_id",
                "last_buy",
                "first_buy",
            ],
            axis=1,
        )

        recs = df

        recs = recs.rename(
            {
                "article_id": f"{self.RECS_NAME}_recs",
                "time_score": f"{self.RECS_NAME}_score",
                "rank_time": f"{self.RECS_NAME}_rank",
            },
            axis=1,
        )
        return recs


if __name__ == "__main__":
    KIND = "train"
    ALPHA = 0.9
    EPS = 1e-6
    CUTOFF = 100

    dataset = StratifiedDataset()

    rec = TimeWeight(kind=KIND, dataset=dataset, cutoff=0)
    rec.eval_recommendations(write_log=False)
    # rec.save_recommendations()
