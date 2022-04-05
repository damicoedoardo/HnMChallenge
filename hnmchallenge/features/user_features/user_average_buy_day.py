from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserFeature


class UserAvgBuyDay(UserFeature):
    FEATURE_NAME = "user_average_buy_day"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_full_data()
        )

        # filter on the last month_data
        df = data_df.drop_duplicates(["t_dat", DEFAULT_USER_COL])

        df["date_diff"] = (
            df.pop("t_dat")
            .groupby(df["customer_id"])
            .diff()
            .dt.days.fillna(0, downcast="infer")
        )

        user_average = df.groupby(DEFAULT_USER_COL)["date_diff"].mean().reset_index()

        feature = user_average
        user_df = self._get_keys_df()
        feature = pd.merge(user_df, feature, on="customer_id", how="left")
        feature = feature.rename({"date_diff": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature
