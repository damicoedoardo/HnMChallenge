from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class SalesFactor(ItemFeature):
    FEATURE_NAME = "sales_factor"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_last_month_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        data_df=data_df[[DEFAULT_ITEM_COL,"price"]]
        data_df["max_price"] = data_df.groupby(DEFAULT_ITEM_COL)["price"].transform("max")
        data_df["sale_factor"] = 1 - (data_df["price"] /data_df["max_price"])
        data_df = data_df[[ DEFAULT_ITEM_COL, "sale_factor"]]
        data_df=data_df.drop_duplicates([DEFAULT_ITEM_COL], keep="last").sort_values(DEFAULT_ITEM_COL)
        data_df=data_df.reset_index()
        feature = data_df[[ DEFAULT_ITEM_COL, "sale_factor"]]
        feature = feature.rename({"sale_factor": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature


if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["train", "full"]:
        feature =SalesFactor(dataset, kind)
        feature.save_feature()