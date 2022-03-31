from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import ItemFeature


class IndexGroupName(ItemFeature):
    FEATURE_NAME = "index_group_name"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dataset.get_articles_df()
        index_code = pd.get_dummies(item_df["index_group_name"])
        item = item_df[DEFAULT_ITEM_COL].to_frame()
        feature = item.join(index_code)
        feature.columns = feature.columns.map(str)

        print(feature)
        return feature
