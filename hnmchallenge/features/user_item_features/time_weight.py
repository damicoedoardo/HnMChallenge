import datetime
from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserItemFeature


class TimeWeight(UserItemFeature):
    FEATURE_NAME = "tdiff"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        data_df = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat", "price"]]
        data_df["tdiff"] = data_df["t_dat"].apply(
            lambda x: 1 / (datetime.datetime(2020, 9, 23) - x).days
        )

        feature = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "tdiff"]]
        feature = feature[
            ~(feature[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].duplicated())
        ].drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep="last")
        feature = feature.rename({"tdiff": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature
