import logging

import numpy as np
import pandas as pd
from hnmchallenge.constant import *

from hnmchallenge.datasets.first_week_dataset import FirstWeekDataset
from hnmchallenge.datasets.second_week_dataset import SecondWeekDataset
from hnmchallenge.datasets.third_week_dataset import ThirdWeekDataset
from hnmchallenge.datasets.fourth_week_dataset import FourthWeekDataset
from hnmchallenge.datasets.fifth_week_dataset import FifthWeekDataset

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
        filter_on_candidates=None,
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
        # this is done when performin local evaluation, if used for the challenge comment else block
        else:
            eval_data = self.dataset.get_evaluation_data()
            users_eval = eval_data[DEFAULT_USER_COL].unique()
            prediction_data = data_df[data_df[DEFAULT_USER_COL].isin(users_eval)]
        # data that you use to compute similarity

        # Using the full data available perform better
        # data_sim = data_df[data_df["t_dat"] > "2020-07-31"]

        # instantiate the recommender algorithm
        recom = ItemKNN(self.dataset, time_weight=self.time_weight, topk=1000)

        print(set_color("Computing similarity...", "green"))
        recom.compute_similarity_matrix(data_df)
        recs = recom.recommend_multicore(
            interactions=prediction_data,
            batch_size=10_000,
            num_cpus=12,
            remove_seen=self.remove_seen,
            white_list_mb_item=None,
            filter_on_candidates=self.filter_on_candidates,
            cutoff=self.cutoff,
            insert_gt=False,
        )
        print(recs)

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
    FC = True

    DATASETS = [
        FirstWeekDataset(),
        # SecondWeekDataset(),
        # ThirdWeekDataset(),
        # FourthWeekDataset(),
        # FifthWeekDataset(),
    ]

    for dataset in DATASETS:
        for kind in ["full"]:  # , "full"]:
            # for kind in ["train", "full"]:
            candidate_items = dataset.get_candidate_items(kind=kind)
            print(len(candidate_items))
            rec_ens = ItemKNNRecs(
                kind=kind,
                cutoff=200,
                time_weight=TW,
                remove_seen=REMOVE_SEEN,
                dataset=dataset,
                filter_on_candidates=candidate_items,
            )

            if kind == "train":
                map_score, recall_score = rec_ens.eval_recommendations()
                print(map_score)
                print(recall_score)
            rec_ens.save_recommendations()
