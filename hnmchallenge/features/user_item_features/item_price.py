from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class ItemPrice(UserItemFeature):
    FEATURE_NAME = "item_price"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        print(self.kind)
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        feature = (
            data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat"]]
            .drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL],keep='last')
            .drop("t_dat", axis=1)
        )
        feature = feature.rename({"price": self.FEATURE_NAME}, axis=1)
        return feature



if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["train", "full"]:
        feature = ItemPrice(dataset, kind)
        feature.save_feature()
