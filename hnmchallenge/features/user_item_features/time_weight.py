from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.feature_manager import UserItemFeature
import datetime
from hnmchallenge.stratified_dataset import StratifiedDataset


class TimeWeight(UserItemFeature):
    FEATURE_NAME = "tdiff"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        data_df = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat", "price"]].drop_duplicates()
        data_df["tdiff"]=data_df['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,23) - x).days)

        feature = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat", "tdiff"]]
        feature = feature[~(feature[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat"]].duplicated())].drop_duplicates()
        feature = feature.rename({"tdiff": self.FEATURE_NAME}, axis=1)
        return feature



if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["train", "full"]:
        feature = TimeWeight(dataset, kind)
        feature.save_feature()