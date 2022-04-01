from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import ItemFeature


class NumberBought(ItemFeature):
    FEATURE_NAME = "number_bought"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        number_bought = (
            data_df.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])["price"]
            .count()
            .rename("number_bought")
            .reset_index()
            .drop([DEFAULT_USER_COL], axis=1)
        )
        count = []
        for i in number_bought[DEFAULT_ITEM_COL]:
            if number_bought["number_bought"][i] > 1:
                count.append(1)
            else:
                count.append(0)
        number_bought["count"] = count
        number_bought = number_bought.drop("number_bought", axis=1)
        feature = (
            number_bought.groupby([DEFAULT_ITEM_COL])["count"]
            .count()
            .rename(self.FEATURE_NAME)
            .reset_index()
        )
        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
