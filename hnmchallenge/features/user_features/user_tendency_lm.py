from unicodedata import name

import numpy as np
import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class UserTendencyLM(UserFeature):
    FEATURE_NAME = "user_tendency_lm"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_last_month_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )

        # filter on the last month_data
        data_df = data_df[data_df["t_dat"] > "2020-08-31"]

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
        merge_df1["user_tendency"] = 1 - (merge_df1["count_y"] / merge_df1["count_x"])
        user_diff = merge_df1.drop(
            ["article_id_x", "count_x", "article_id_y", "count_y"], axis=1
        )
        feature = user_diff.reset_index()
        user_df = self._get_keys_df()
        feature = pd.merge(user_df, feature, on="customer_id", how="left")
        feature = feature.rename({"user_tendency": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature


if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["full", "train"]:
        feature = UserTendencyLM(dataset, kind)
        feature.save_feature()
