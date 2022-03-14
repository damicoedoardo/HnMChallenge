from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.feature_manager import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class ProductGroupName(ItemFeature):
    FEATURE_NAME = "product_group_name"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dr.get_filtered_articles()
        pgn = pd.get_dummies(item_df["product_group_name"])
        item = item_df[DEFAULT_ITEM_COL].to_frame()
        feature = item.join(pgn)
        print(feature)
        return feature


if __name__ == "__main__":
    dataset = StratifiedDataset()
    feature = ProductGroupName(dataset, kind="full")
    feature.save_feature()
