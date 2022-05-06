import datetime
from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserItemFeature


class TimeScore(UserItemFeature):
    """ """

    FEATURE_NAME = "time_score"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        data_df = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat", "price"]]

        data_df["last_buy"] = data_df.groupby(DEFAULT_USER_COL)["t_dat"].transform(max)
        data_df["first_buy"] = data_df.groupby(DEFAULT_USER_COL)["t_dat"].transform(min)

        data_df["time_score"] = (data_df["t_dat"] - data_df["first_buy"]) / (
            data_df["last_buy"] - data_df["first_buy"]
        )

        # data_df["time_score"] = data_df["time_score"].fillna(0)

        data_df.drop(["last_buy", "first_buy"], axis=1)

        feature = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "time_score"]]

        feature = feature[
            ~(feature[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].duplicated())
        ].drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep="last")

        feature = feature.rename({"time_score": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature
