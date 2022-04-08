from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class PopSales1(ItemFeature):
    FEATURE_NAME = "popularity_sales_channel_1"

    def __init__(self, dataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        data_df = (
            self.dataset.get_holdin()
            if self.kind == "train"
            else self.dataset.get_full_data()
        )
        duplicated_rows = data_df[
            data_df.duplicated(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        ]
        duplicated_rows = duplicated_rows[duplicated_rows["sales_channel_id"] == 1]
        count_mb = duplicated_rows.groupby(DEFAULT_ITEM_COL).count()
        feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
            columns={"t_dat": self.FEATURE_NAME}
        )

        # normalisation popularity
        feature["popularity_sales_channel_1"] = (
            feature["popularity_sales_channel_1"]
            - feature["popularity_sales_channel_1"].min()
        ) / (
            feature["popularity_sales_channel_1"].max()
            - feature["popularity_sales_channel_1"].min()
        )

        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
