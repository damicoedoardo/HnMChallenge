from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class ItemCountLastMonth(ItemFeature):
    FEATURE_NAME = "item_count_last_month"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dr.get_filtered_full_data()
            if self.kind == "train"
            else self.dataset.get_holdin()
        )
        data_df=data_df[data_df["t_dat"]>="2020-09-01"]
        duplicated_rows = data_df[data_df.duplicated(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])]
        count_mb = duplicated_rows.groupby(DEFAULT_ITEM_COL).count()
        feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
            columns={"t_dat": "count_last_month"}
        )
        item_df=self._get_keys_df()
        feature=pd.merge(item_df, feature, on=DEFAULT_ITEM_COL,how='left')
        return feature


if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["full", "train"]:
        feature = ItemCountLastMonth(dataset, kind)
        feature.save_feature()