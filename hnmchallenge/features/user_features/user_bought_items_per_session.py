from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserFeature


class UserAvgBuySession(UserFeature):
    FEATURE_NAME = "user_average_buy_day"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )

        # filter on the last month_data
        fd1 = (
            data_df.groupby([DEFAULT_USER_COL, "t_dat"])
            .size()
            .reset_index(name="counts")
        )
        feature = (
            fd1.groupby([DEFAULT_USER_COL])["counts"].mean().reset_index(name="average")
        )

        user_df = self._get_keys_df()
        feature = pd.merge(user_df, feature, on="customer_id", how="left")
        feature = feature.rename({"date_diff": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature
