from unicodedata import name

import numpy as np
import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class PopularityCumulativeMultipleBuy(ItemFeature):
    FEATURE_NAME = "popularity_cumulative_multiple_buy"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        max_date = data_df["t_dat"].max()
        intervals = [
            (max_date - pd.to_timedelta(7, unit="D"), max_date),
            (max_date - pd.to_timedelta(14, unit="D"), max_date),
            (max_date - pd.to_timedelta(21, unit="D"), max_date),
            (max_date - pd.to_timedelta(28, unit="D"), max_date),
        ]
        fd = data_df
        df = pd.DataFrame(columns=[DEFAULT_ITEM_COL])
        for idx, i in enumerate(intervals):
            m = np.logical_or.reduce(
                [np.logical_and(fd["t_dat"] >= i[0], fd["t_dat"] <= i[1])]
            )
            data_df = fd.loc[m]
            duplicated_rows = data_df[
                data_df.duplicated(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
            ]
            count_mb = duplicated_rows.groupby(DEFAULT_ITEM_COL).count()
            feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
                columns={"t_dat": f"popularity_{idx+1}_multiple_buy"}
            )
            feature[f"popularity_{idx+1}_multiple_buy"] = (
                feature[f"popularity_{idx+1}_multiple_buy"]
                - feature[f"popularity_{idx+1}_multiple_buy"].min()
            ) / (
                feature[f"popularity_{idx+1}_multiple_buy"].max()
                - feature[f"popularity_{idx+1}_multiple_buy"].min()
            )
            df = df.merge(feature, on=DEFAULT_ITEM_COL, how="outer")
        feature = df
        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
