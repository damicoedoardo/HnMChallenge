from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.feature_manager import UserItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class SalesChannel(UserItemFeature):
    FEATURE_NAME = "sales_channel_id"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        feature = (
            data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL,"t_dat", "sales_channel_id"]]
            .drop_duplicates()
            
        )
        feature = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat", "sales_channel_id"]]
        feature = feature[~(feature[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat"]].duplicated())].drop_duplicates()
        feature = feature.rename({"sales_channel_id": self.FEATURE_NAME}, axis=1)
        return feature



if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["train", "full"]:
        feature = SalesChannel(dataset, kind)
        feature.save_feature()
