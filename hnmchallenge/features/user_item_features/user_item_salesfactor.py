from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserItemFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class UserItemSalesFactor(UserItemFeature):
    FEATURE_NAME = "sale_factor"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        data_df = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat", "price"]]
        data_df["max_price"] = data_df.groupby(DEFAULT_ITEM_COL)["price"].transform(
            "max"
        )
        data_df["sale_factor"] = 1 - (data_df["price"] / data_df["max_price"])

        feature = data_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "sale_factor"]]
        feature = feature[
            ~(feature[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].duplicated())
        ].drop_duplicates([DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep="last")
        feature = feature.rename({"sale_factor": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature


if __name__ == "__main__":
    dataset = StratifiedDataset()
    for kind in ["train", "full"]:
        feature = UserItemSalesFactor(dataset, kind)
        feature.save_feature()
