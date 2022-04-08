from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class ItemPriceProduct(ItemFeature):
    FEATURE_NAME = "item_price_stat_product"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        articles = self.dataset.get_articles_df()
        articles = articles[[DEFAULT_ITEM_COL, "product_type_no"]]
        fd2 = pd.merge(data_df, articles, on=DEFAULT_ITEM_COL, how="left")
        fd3 = (
            fd2.groupby([DEFAULT_ITEM_COL, "product_type_no"])["price"]
            .mean()
            .reset_index(name="mean")
        )
        fd3 = pd.merge(fd2, fd3, on=[DEFAULT_ITEM_COL, "product_type_no"], how="left")
        fd4 = (
            fd2.groupby([DEFAULT_ITEM_COL, "product_type_no"])["price"]
            .min()
            .reset_index(name="min")
        )
        fd4 = pd.merge(fd3, fd4, on=[DEFAULT_ITEM_COL, "product_type_no"], how="left")
        fd5 = (
            fd2.groupby([DEFAULT_ITEM_COL, "product_type_no"])["price"]
            .max()
            .reset_index(name="max")
        )
        fd5 = pd.merge(fd4, fd5, on=[DEFAULT_ITEM_COL, "product_type_no"], how="left")
        fd5["price_mean"] = fd5["price"] / fd5["mean"]
        fd5["price_min"] = fd5["price"] / fd5["min"]
        fd5["price_max"] = fd5["price"] / fd5["max"]
        feature = fd5[[DEFAULT_ITEM_COL, "price_mean", "price_min", "price_max"]]
        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
