from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class IndexGroupNO(ItemFeature):
    FEATURE_NAME = "index_group_number"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dr.get_filtered_articles()
        feature = item_df[[DEFAULT_ITEM_COL, "index_group_no"]]
        print(feature)
        return feature


if __name__ == "__main__":
    for kind in ["train", "full"]:
        dataset = StratifiedDataset()
        feature = IndexGroupNO(dataset, kind=kind)
        feature.save_feature()
