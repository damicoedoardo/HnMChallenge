from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class ItemAgeDescribe(ItemFeature):
    FEATURE_NAME = "item_age_describe"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        customers = self.dataset.get_customers_df()
        customers = customers[[DEFAULT_USER_COL, "age"]]
        fd2 = pd.merge(data_df, customers, on=DEFAULT_USER_COL, how="left")
        fd2 = fd2.drop([DEFAULT_USER_COL, "price", "sales_channel_id", "t_dat"], axis=1)
        fd3 = fd2.groupby(DEFAULT_ITEM_COL)["age"].describe()
        feature = fd3.rename(
            columns={
                "count": "age_count",
                "mean": "age_mean",
                "std": "age_std",
                "min": "age_min",
                "max": "age_max",
                "25%": "age_25%",
                "50%": "age_50%",
                "75%": "age_75%",
            },
            inplace=False,
        ).reset_index()

        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
