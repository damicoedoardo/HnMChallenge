from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.dataset import Dataset
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class GraphicalAppearanceNO(ItemFeature):
    FEATURE_NAME = "graphical_appearance_no"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        item_df = self.dr.get_full_articles()
        feature = item_df[[DEFAULT_ITEM_COL, "graphical_appearance_no"]]
        # gan = pd.get_dummies(item_df["graphical_appearance_no"])
        # item = item_df[DEFAULT_ITEM_COL].to_frame()
        # feature = item.join(gan)
        # feature.columns = feature.columns.map(str)

        print(feature)
        return feature


if __name__ == "__main__":
    for kind in ["train", "full"]:
        dataset = Dataset()
        feature = GraphicalAppearanceNO(dataset, kind=kind)
        feature.save_feature()
