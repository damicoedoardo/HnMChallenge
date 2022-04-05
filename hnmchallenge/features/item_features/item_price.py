from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class Price(ItemFeature):
    FEATURE_NAME = "price"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )

        # take the price on the last available day
        last_date_item = (
            data_df.groupby(DEFAULT_ITEM_COL)[["t_dat"]].max().reset_index()
        )
        feature = pd.merge(
            data_df, last_date_item, on=[DEFAULT_ITEM_COL, "t_dat"]
        ).drop_duplicates(subset=DEFAULT_ITEM_COL)[[DEFAULT_ITEM_COL, "price"]]

        # compute the average price
        avg_price = (
            data_df.groupby(DEFAULT_ITEM_COL)["price"]
            .mean()
            .reset_index()
            .rename(columns={"price": "avg_price"})
        )
        feature = pd.merge(feature, avg_price, on=DEFAULT_ITEM_COL)

        # compute the average price
        max_price = (
            data_df.groupby(DEFAULT_ITEM_COL)["price"]
            .max()
            .reset_index()
            .rename(columns={"price": "max_price"})
        )
        feature = pd.merge(feature, max_price, on=DEFAULT_ITEM_COL)

        min_price = (
            data_df.groupby(DEFAULT_ITEM_COL)["price"]
            .min()
            .reset_index()
            .rename(columns={"price": "min_price"})
        )
        feature = pd.merge(feature, min_price, on=DEFAULT_ITEM_COL)

        std_price = (
            data_df.groupby(DEFAULT_ITEM_COL)["price"]
            .std()
            .reset_index()
            .rename(columns={"price": "std_price"})
        )
        feature = pd.merge(feature, std_price, on=DEFAULT_ITEM_COL)

        # AAYUSH veraion
        # data_df = data_df[[DEFAULT_ITEM_COL, "price"]]
        # data_df = data_df.drop_duplicates([DEFAULT_ITEM_COL], keep="last").sort_values(
        #     DEFAULT_ITEM_COL
        # )
        # data_df = data_df.reset_index()
        # feature = data_df[[DEFAULT_ITEM_COL, "price"]]
        # feature = feature.rename({"price": self.FEATURE_NAME}, axis=1)

        # Some item have been bought only in the last week
        keys_df = self._get_keys_df()
        feature = pd.merge(keys_df, feature, on=DEFAULT_ITEM_COL, how="left")

        print(feature)
        return feature
