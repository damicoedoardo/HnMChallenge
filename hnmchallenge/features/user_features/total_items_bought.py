from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import UserFeature


class TotalItemsBought(UserFeature):
    FEATURE_NAME = "total_items_bought"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )

        # filter on the last month_data
        df = data_df.drop_duplicates(["t_dat", DEFAULT_USER_COL])

        df = df.groupby([DEFAULT_USER_COL]).count().reset_index()
        feature = df[[DEFAULT_USER_COL, "price"]]
        user_df = self._get_keys_df()
        feature = pd.merge(user_df, feature, on="customer_id", how="left")
        feature = feature.rename({"price": self.FEATURE_NAME}, axis=1)
        print(feature)
        return feature
