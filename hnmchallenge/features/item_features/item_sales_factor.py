from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class SalesFactor(ItemFeature):
    FEATURE_NAME = "sales_factor"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        data_df["max_price"] = data_df.groupby(DEFAULT_ITEM_COL)["price"].transform(
            "max"
        )
        data_df["sale_factor"] = 1 - (data_df["price"] / data_df["max_price"])
        # data_df = data_df[[DEFAULT_ITEM_COL, "sale_factor"]]
        # edo added sorting
        data_df = (
            data_df.sort_values("t_dat")
            .drop_duplicates([DEFAULT_ITEM_COL], keep="last")
            .sort_values(DEFAULT_ITEM_COL)
        )
        data_df = data_df.reset_index()
        feature = data_df[[DEFAULT_ITEM_COL, "sale_factor"]]
        feature = feature.rename({"sale_factor": self.FEATURE_NAME}, axis=1)

        # Some item have been bought only in the last week
        keys_df = self._get_keys_df()
        feature = pd.merge(keys_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
