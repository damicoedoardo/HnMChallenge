import logging

import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color


class TimePop(RecsInterface):
    def __init__(
        self,
        kind: str,
        dataset: StratifiedDataset,
        alpha: float = 0.95,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(kind, dataset)
        self.alpha = alpha
        self.eps = eps

        # set recommender name
        self.RECS_NAME = f"TimePop_alpha_{alpha}_eps_{eps}"

    def get_recommendations(self, cutoff: int = 100) -> pd.DataFrame:
        data_df = (
            self.dataset.get_last_month_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        # data that you use to compute similarity
        # Using the full data available perform better
        data_sim = data_df[data_df["t_dat"] > "2020-08-31"]

        data_sim = data_sim.drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        count_mb = data_sim.groupby(DEFAULT_ITEM_COL).count()
        feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
            columns={"t_dat": "popularity"}
        )
        feature["popularity_score"] = (
            feature["popularity"] - feature["popularity"].min()
        ) / (feature["popularity"].max() - feature["popularity"].min())
        feature["rank"] = (
            feature["popularity_score"].rank(ascending=False, method="min").astype(int)
        )
        feature_k = feature[feature["rank"] <= cutoff]
        feature_k["temp"] = 1

        df = data_sim
        df["last_buy"] = df.groupby(DEFAULT_USER_COL)["t_dat"].transform(max)
        df["first_buy"] = df.groupby(DEFAULT_USER_COL)["t_dat"].transform(min)
        df["time_score"] = (df["t_dat"] - df["first_buy"]) / (
            df["last_buy"] - df["first_buy"]
        )
        df = df.fillna(1)
        df["rank_time"] = (
            df.groupby(DEFAULT_USER_COL)["time_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        final = pd.merge(df, feature, on=DEFAULT_ITEM_COL, how="left")

        user = data_sim
        user = user.drop_duplicates([DEFAULT_USER_COL])
        user["temp"] = 1
        user = user[[DEFAULT_USER_COL, "temp"]]
        final1 = pd.merge(user, feature_k, on="temp")
        final1 = final1.drop("temp", axis=1)
        final = final.drop(
            [
                "popularity_score",
                "rank",
                "popularity",
                "t_dat",
                "price",
                "sales_channel_id",
                "last_buy",
                "first_buy",
            ],
            axis=1,
        )
        final2 = pd.merge(
            final, final1, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="outer"
        )
        final2 = final2.fillna(0)
        final2 = final2.drop(["rank_time", "rank"], axis=1)
        final2["score"] = (
            self.alpha * final2["time_score"]
            + (1 - self.alpha + self.eps) * final2["popularity_score"]
        )
        final2["rank"] = (
            final2.groupby(DEFAULT_USER_COL)["score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        final2 = final2.drop(["time_score", "popularity", "popularity_score"], axis=1)
        recs = final2

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
    ALPHA = 0.5
    EPS = 1e-6
    dataset = StratifiedDataset()

    rec_ens = TimePop(kind=KIND, alpha=ALPHA, eps=EPS, dataset=dataset)
    rec_ens.eval_recommendations(cutoff=100, write_log=False)
    # rec_ens.save_recommendations()
