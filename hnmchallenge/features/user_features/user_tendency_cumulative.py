from unicodedata import name

import numpy as np
import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserFeature


class UserTendencyCumulative(UserFeature):
    FEATURE_NAME = "user_tendency_cumulative"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )

        # filter on the last month_data
        max_date = data_df["t_dat"].max()
        df = pd.DataFrame(columns=[DEFAULT_USER_COL])
        intervals = [
            (max_date - pd.to_timedelta(7, unit="D"), max_date),
            (max_date - pd.to_timedelta(14, unit="D"), max_date),
            (max_date - pd.to_timedelta(21, unit="D"), max_date),
            (max_date - pd.to_timedelta(28, unit="D"), max_date),
        ]
        fd = data_df

        df = pd.DataFrame(columns=[DEFAULT_USER_COL])
        for idx, i in enumerate(intervals):
            m = np.logical_or.reduce(
                [np.logical_and(fd["t_dat"] >= i[0], fd["t_dat"] <= i[1])]
            )
            data_df = fd.loc[m]
            item_per_user1 = data_df.groupby("customer_id")["article_id"].apply(list)
            unique_item_per_user1 = item_per_user1.apply(np.unique)
            df_multiple1 = item_per_user1.to_frame()
            df_unique1 = unique_item_per_user1.to_frame()
            df_multiple1["count"] = df_multiple1.apply(
                lambda row: len(row["article_id"]), axis=1
            )
            df_unique1["count"] = df_unique1.apply(
                lambda row: len(row["article_id"]), axis=1
            )
            merge_df1 = pd.merge(df_multiple1, df_unique1, on="customer_id")
            merge_df1[f"user_tendency_{idx+1}"] = 1 - (
                merge_df1["count_y"] / merge_df1["count_x"]
            )
            user_diff = merge_df1.drop(
                ["article_id_x", "count_x", "article_id_y", "count_y"], axis=1
            )
            df = df.merge(user_diff, on=DEFAULT_USER_COL, how="outer")
        user_df = self._get_keys_df()
        feature = pd.merge(user_df, df, on="customer_id", how="left")
        print(feature)
        return feature
