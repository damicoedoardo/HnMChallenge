from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserFeature


class AvgPrice(UserFeature):
    FEATURE_NAME = "avg_price"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        data_df = data_df.groupby(DEFAULT_USER_COL).mean().reset_index()
        feature = data_df[[DEFAULT_USER_COL, "price"]]
        feature = feature.rename({"price": self.FEATURE_NAME}, axis=1)

        # Losing the users that have bought something for the first time on the last week
        keys_df = self._get_keys_df()
        feature = pd.merge(keys_df, feature, on=DEFAULT_USER_COL, how="left")
        print(feature)
        return feature
