from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class GarmentGroupName(ItemFeature):
    FEATURE_NAME = "garment_group_name"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dataset.get_articles_df()
        pgn = pd.get_dummies(item_df["garment_group_name"])
        item = item_df[DEFAULT_ITEM_COL].to_frame()
        feature = item.join(pgn)
        print(feature)
        return feature
