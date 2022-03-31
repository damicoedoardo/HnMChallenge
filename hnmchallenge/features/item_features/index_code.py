from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.dataset import Dataset
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class IndexCode(ItemFeature):
    FEATURE_NAME = "index_code"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dr.get_full_articles()
        index_code = pd.get_dummies(item_df["index_code"])
        item = item_df[DEFAULT_ITEM_COL].to_frame()
        feature = item.join(index_code)
        feature.columns = feature.columns.map(str)

        print(feature)
        return feature


if __name__ == "__main__":
    dataset = Dataset()
    feature = IndexCode(dataset, kind="full")
    feature.save_feature()
