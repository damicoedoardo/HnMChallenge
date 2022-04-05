from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import ItemFeature


class GraphicalAppearanceNOGBM(ItemFeature):
    FEATURE_NAME = "graphical_appearance_no_gbm"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dataset.get_articles_df()
        feature = item_df[[DEFAULT_ITEM_COL, "graphical_appearance_no"]]

        sequential = (
            pd.Series(feature["graphical_appearance_no"].unique())
            .reset_index()
            .rename(columns={0: "graphical_appearance_no"})
        )
        feature = feature.merge(sequential, on="graphical_appearance_no")
        feature = feature.rename({"index": self.FEATURE_NAME}, axis=1)
        feature = feature.drop("graphical_appearance_no", axis=1)

        print(feature)
        return feature
