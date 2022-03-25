import logging

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color


class PopularityRecs(RecsInterface):
    def __init__(
        self,
        kind: str,
        dataset: StratifiedDataset,
        cutoff: int = 100,
    ) -> None:
        super().__init__(kind, dataset, cutoff)

        # set recommender name
        self.RECS_NAME = f"Popularity_cutoff_{cutoff}"

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

        count_mb = last_month_data.groupby(DEFAULT_ITEM_COL).count()

        feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
            columns={"t_dat": "popularity"}
        )

        feature["popularity_score"] = (
            feature["popularity"] - feature["popularity"].min()
        ) / (feature["popularity"].max() - feature["popularity"].min())

        feature["rank"] = (
            feature["popularity_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        feature_k = feature[feature["rank"] <= self.cutoff]
        feature_k["temp"] = 1

        user = data_df
        user = user.drop_duplicates([DEFAULT_USER_COL])
        user["temp"] = 1
        user = user[[DEFAULT_USER_COL, "temp"]]

        final1 = pd.merge(user, feature_k, on="temp")
        final1 = final1.drop("temp", axis=1)
        final1 = final1.drop(
            [
                "popularity",
            ],
            axis=1,
        )

        recs = final1

        recs = recs.rename(
            {
                "article_id": f"{self.RECS_NAME}_recs",
                "popularity_score": f"{self.RECS_NAME}_score",
                "rank": f"{self.RECS_NAME}_rank",
            },
            axis=1,
        )
        return recs


if __name__ == "__main__":
    KIND = "full"
    ALPHA = 0.9
    EPS = 1e-6
    CUTOFF = 100

    dataset = StratifiedDataset()

    rec = PopularityRecs(kind=KIND, dataset=dataset, cutoff=CUTOFF)
    # rec.eval_recommendations(write_log=False)
    rec.save_recommendations()
