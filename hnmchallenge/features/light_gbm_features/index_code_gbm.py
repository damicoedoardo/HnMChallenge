from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import ItemFeature


class IndexCodeGBM(ItemFeature):
    FEATURE_NAME = "index_code_gbm"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dataset.get_articles_df()
        feature = item_df[[DEFAULT_ITEM_COL, "index_code"]]
        sequential = (
            pd.Series(feature["index_code"].unique())
            .reset_index()
            .rename(columns={0: "index_code"})
        )
        feature = feature.merge(sequential, on="index_code")
        feature = feature.rename({"index": self.FEATURE_NAME}, axis=1)
        feature = feature.drop("index_code", axis=1)

        print(feature)
        return feature
