from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class ItemCount(ItemFeature):
    FEATURE_NAME = "popularity"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_last_month_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
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
    dataset = StratifiedDataset()
    for kind in ["full", "train"]:
        feature = ItemCount(dataset, kind)
        feature.save_feature()
