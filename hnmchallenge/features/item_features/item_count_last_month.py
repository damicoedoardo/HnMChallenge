from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.dataset import Dataset
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class ItemCountLastMonth(ItemFeature):
    FEATURE_NAME = "popularity_last_month"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_full_data()
        )
        data_df = data_df[data_df["t_dat"] >= "2020-09-1"]
        duplicated_rows = data_df[
            data_df.duplicated(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        ]
        count_mb = duplicated_rows.groupby(DEFAULT_ITEM_COL).count()
        feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
            columns={"t_dat": self.FEATURE_NAME}
        )
        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature


if __name__ == "__main__":
    dataset = Dataset()
    for kind in ["full", "train"]:
        feature = ItemCountLastMonth(dataset, kind)
        feature.save_feature()
