from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import ItemFeature


class ProductGroupNameGBM(ItemFeature):
    FEATURE_NAME = "product_group_name_gbm"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dataset.get_articles_df()
        feature = item_df[[DEFAULT_ITEM_COL, "product_group_name"]]
        sequential = (
            pd.Series(feature["product_group_name"].unique())
            .reset_index()
            .rename(columns={0: "product_group_name"})
        )
        feature = feature.merge(sequential, on="product_group_name")
        feature = feature.rename({"index": self.FEATURE_NAME}, axis=1)
        feature = feature.drop("product_group_name", axis=1)

        print(feature)
        return feature
