import logging

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.datasets.last2month_last_week import L2MLWDataset
from hnmchallenge.datasets.last_month_last_4day import LML4DDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models.userknn.userknn import UserKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.utils.logger import set_color


class UserKNNRecs(RecsInterface):
    def __init__(
        self,
        kind: str,
        dataset,
        time_weight: bool = True,
        remove_seen: bool = False,
        cutoff: int = 100,
    ) -> None:
        super().__init__(kind, dataset, cutoff)
        self.time_weight = time_weight
        self.remove_seen = remove_seen

        # set recommender name
        self.RECS_NAME = f"UserKNN_tw_{time_weight}_rs_{remove_seen}"

    def get_recommendations(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        # data that you use to compute similarity

        # Using the full data available perform better
        # data_sim = data_df[data_df["t_dat"] > "2020-09-01"]

        # instantiate the recommender algorithm
        recom = UserKNN(self.dataset, time_weight=self.time_weight, topk=1000)

        print(set_color("Computing similarity...", "green"))
        recom.compute_similarity_matrix(data_df)
        recs = recom.recommend_similaripy(
            interactions=data_df, cutoff=self.cutoff, remove_seen=self.remove_seen
        )

        recs = recs.rename(
            {
                "article_id": f"{self.RECS_NAME}_recs",
                "prediction": f"{self.RECS_NAME}_score",
                "rank": f"{self.RECS_NAME}_rank",
            },
            axis=1,
        )
        return recs


if __name__ == "__main__":
    TW = False
    REMOVE_SEEN = False
    dataset = LMLWDataset()

    for kind in ["train", "full"]:
        rec_ens = UserKNNRecs(
            kind=kind,
            cutoff=100,
            time_weight=TW,
            remove_seen=REMOVE_SEEN,
            dataset=dataset,
        )
        rec_ens.eval_recommendations()
        # rec_ens.save_recommendations()
