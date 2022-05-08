from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserFeature


class LastBuyDate(UserFeature):
    FEATURE_NAME = "last_buy_date"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        fd = self.dataset.get_holdout()
        max_date = fd["t_dat"].max() + pd.to_timedelta(1, unit="D")
        data = data_df.groupby([DEFAULT_USER_COL])["t_dat"].max().reset_index()
        data["t_diff"] = data["t_dat"].apply(lambda x: 1 / (max_date - x).days)
        feature = data[[DEFAULT_USER_COL, "t_diff"]]
        # Losing the users that have bought something for the first time on the last week
        keys_df = self._get_keys_df()
        feature = pd.merge(keys_df, feature, on=DEFAULT_USER_COL, how="left")
        print(feature)
        return feature
