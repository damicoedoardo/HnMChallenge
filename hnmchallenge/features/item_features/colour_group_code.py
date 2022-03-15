from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.feature_manager import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class ColourGroupCode(ItemFeature):
    FEATURE_NAME = "colour_group_code"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        feature = self.dr.get_filtered_articles()[
            [DEFAULT_ITEM_COL, "colour_group_code"]
        ]
        print(feature)
        return feature


if __name__ == "__main__":
    dataset = StratifiedDataset()
    feature = ColourGroupCode(dataset, kind="full")
    feature.save_feature()