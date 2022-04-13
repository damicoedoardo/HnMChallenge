import logging

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.models.ease.ease import EASE
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.utils.logger import set_color


class EaseRecs(RecsInterface):
    def __init__(
        self,
        l2: float,
        kind: str,
        dataset,
        time_weight: bool = True,
        remove_seen: bool = False,
        cutoff: int = 100,
    ) -> None:
        super().__init__(kind, dataset, cutoff)
        self.time_weight = time_weight
        self.remove_seen = remove_seen
        self.l2 = l2

        # set recommender name
        self.RECS_NAME = f"EASE_tw_{time_weight}_rs_{remove_seen}_l2_{l2}"

    def get_recommendations(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )

        if self.kind == "train":
            # retrieve the holdout
            holdout = self.dataset.get_holdout()
            users_holdout = holdout[DEFAULT_USER_COL].unique()
            prediction_data = data_df[data_df[DEFAULT_USER_COL].isin(users_holdout)]
        else:
            prediction_data = data_df

        # data that you use to compute similarity
        # Using the full data available perform better
        data_sim = data_df[data_df["t_dat"] > "2020-08-31"]

        # instantiate the recommender algorithm
        recom = EASE(self.dataset, time_weight=self.time_weight, l2=self.l2)

        ######
        # this has to be changed only for debugging purposes
        # ppp = data_df[data_df[DEFAULT_USER_COL].isin(np.arange(100))]

        print(set_color("Computing similarity...", "green"))
        recom.compute_similarity_matrix(data_df)
        recs = recom.recommend_multicore(
            interactions=prediction_data,
            batch_size=40_000,
            num_cpus=72,
            remove_seen=self.remove_seen,
            white_list_mb_item=None,
            cutoff=self.cutoff,
        )

        # recs_list = recs.groupby(DEFAULT_USER_COL)["article_id"].apply(list).to_frame()
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
    KIND = "train"
    TW = True
    REMOVE_SEEN = False
    L2 = 1e-3
    dataset = LMLDDataset()

    for kind in ["train", "full"]:
        ease_rec = EaseRecs(
            kind=kind,
            cutoff=200,
            time_weight=TW,
            remove_seen=REMOVE_SEEN,
            dataset=dataset,
            l2=L2,
        )
        ease_rec.eval_recommendations()
        # ease_rec.save_recommendations()
