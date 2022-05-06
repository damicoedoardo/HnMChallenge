import logging

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.datasets.all_items_last_mont__last_day_last_week import AILMLDWDataset
from hnmchallenge.datasets.all_items_last_month_last_day import AILMLDDataset
from hnmchallenge.datasets.all_items_last_month_last_day_last_2nd_week import (
    AILMLD2WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_day_last_3rd_week import (
    AILMLD3WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_day_last_4th_week import (
    AILMLD4WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_day_last_5th_week import (
    AILMLD5WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_week import AILMLWDataset
from hnmchallenge.datasets.last2month_last_day import L2MLDDataset
from hnmchallenge.datasets.last_2week_last_day import L2WLDDataset
from hnmchallenge.datasets.last_month_last_2nd_week_dataset import LML2WDataset
from hnmchallenge.datasets.last_month_last_3rd_week_dataset import LML3WDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_day_aug_sep import LMLASDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_month_last_week_user import LMLUWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.utils.logger import set_color


class ItemKNNRecs(RecsInterface):
    def __init__(
        self,
        kind: str,
        dataset,
        time_weight: bool = True,
        remove_seen: bool = False,
        cutoff: int = 200,
        filter_on_candidates: bool = False,
    ) -> None:
        super().__init__(kind, dataset, cutoff)
        self.time_weight = time_weight
        self.remove_seen = remove_seen
        self.filter_on_candidates = filter_on_candidates

        # set recommender name
        self.RECS_NAME = f"ItemKNN_tw_{time_weight}_rs_{remove_seen}"

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
        # data_sim = data_df[data_df["t_dat"] > "2020-07-31"]

        # instantiate the recommender algorithm
        recom = ItemKNN(self.dataset, time_weight=self.time_weight, topk=1000)

        print(set_color("Computing similarity...", "green"))
        recom.compute_similarity_matrix(data_df)
        recs = recom.recommend_multicore(
            interactions=prediction_data,
            batch_size=20_000,
            num_cpus=16,
            remove_seen=self.remove_seen,
            white_list_mb_item=None,
            filter_on_candidates=self.filter_on_candidates,
            cutoff=self.cutoff,
            insert_gt=False,
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
    TW = True
    REMOVE_SEEN = False
    FC = False
    # dataset = AILMLD5WDataset()
    dataset = LMLDDataset()
    for kind in ["train", "full"]:
        # for kind in ["train", "full"]:
        rec_ens = ItemKNNRecs(
            kind=kind,
            cutoff=200,
            time_weight=TW,
            remove_seen=REMOVE_SEEN,
            dataset=dataset,
            filter_on_candidates=FC,
        )
        # rec_ens.eval_recommendations()
        rec_ens.save_recommendations()
