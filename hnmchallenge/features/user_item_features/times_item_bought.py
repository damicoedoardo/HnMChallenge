from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class TimesItemBought(UserItemFeature):
    FEATURE_NAME = "times_item_bought"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        feature = data_df.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])["price"].count().rename("number_bought").reset_index()
        feature = feature[~(feature[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].duplicated())].drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL],keep='last')
        feature = feature.rename({"number_bought": self.FEATURE_NAME}, axis=1)
        return feature



if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["train", "full"]:
        feature = TimesItemBought(dataset, kind)
        feature.save_feature()