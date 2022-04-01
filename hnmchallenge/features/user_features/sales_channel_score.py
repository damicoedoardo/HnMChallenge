from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserFeature


class SaleChannelScore(UserFeature):
    FEATURE_NAME = "sales_channel_score"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_full_data()
        )
        count_mb = (
            data_df.groupby(DEFAULT_USER_COL)["sales_channel_id"].size().reset_index()
        )
        df = data_df[data_df["sales_channel_id"] == 2]
        count_mb1 = (
            df.groupby(DEFAULT_USER_COL)["sales_channel_id"].size().reset_index()
        )
        count = pd.merge(count_mb, count_mb1, on=DEFAULT_USER_COL, how="left")
        count = count.fillna(0)
        count["sales_channel_score"] = (
            count["sales_channel_id_y"] / count["sales_channel_id_x"]
        )
        feature = count[[DEFAULT_USER_COL, "sales_channel_score"]]
        feature.fillna(0)
        feature = feature.rename({"sales_channel_score": self.FEATURE_NAME}, axis=1)

        # Losing the users that have bought something for the first time on the last week
        keys_df = self._get_keys_df()
        feature = pd.merge(keys_df, feature, on=DEFAULT_USER_COL, how="left")
        return feature
