import datetime
import logging
from unicodedata import name

import numpy as np
import pandas as pd
from dotenv import main
from hnmchallenge.constant import *
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.datasets.last2month_last_day import L2MLDDataset
from hnmchallenge.datasets.last_2week_last_day import L2WLDDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_day_aug_sep import LMLASDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_month_last_week_user import LMLUWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.features.feature_interfaces import UserItemFeature
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.utils.logger import set_color


class ItemKNNScore(UserItemFeature):
    FEATURE_NAME = "ItemKNN_score"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _check_integrity(self, feature: pd.DataFrame) -> None:
        print("No check integrity for this feature...")

    def _create_feature(self) -> pd.DataFrame:
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

        # instantiate the recommender algorithm
        recom = ItemKNN(self.dataset, time_weight=True, topk=1000)

        print(set_color("Computing similarity...", "green"))
        recom.compute_similarity_matrix(data_df)
        recs = recom.recommend_multicore(
            interactions=prediction_data,
            batch_size=40_000,
            num_cpus=72,
            remove_seen=False,
            white_list_mb_item=None,
            cutoff=1000,
        )
        recs = recs.rename(
            {
                "prediction": f"{recom.name}_score",
                "rank": f"{recom.name}_rank",
            },
            axis=1,
        )
        return recs


if __name__ == "__main__":
    dataset = LMLDDataset()
    feature = ItemKNNScore(dataset=dataset, kind="train")
    feature.save_feature()
