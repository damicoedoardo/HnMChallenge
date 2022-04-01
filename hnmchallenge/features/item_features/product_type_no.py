from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import ItemFeature


class ProductTypeNO(ItemFeature):
    FEATURE_NAME = "product_type_no"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        feature = self.dataset.get_articles_df()[[DEFAULT_ITEM_COL, "product_type_no"]]
        print(feature)
        return feature
