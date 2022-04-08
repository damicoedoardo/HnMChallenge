from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.features.feature_interfaces import ItemFeature


class ItemAgePop(ItemFeature):
    FEATURE_NAME = "item_age_pop"

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

        fd3 = (
            fd2[fd2["age"] <= 25]
            .groupby([DEFAULT_ITEM_COL, "age"])
            .count()
            .reset_index()
        )
        fd3 = fd3[[DEFAULT_ITEM_COL, "age", "t_dat"]].rename(
            {"t_dat": "pop_25"}, axis=1
        )
        fd3["pop_25"] = (fd3["pop_25"] - fd3["pop_25"].min()) / (
            fd3["pop_25"].max() - fd3["pop_25"].min()
        )

        fd4 = (
            fd2[(fd2["age"] > 25) & (fd2["age"] <= 40)]
            .groupby([DEFAULT_ITEM_COL, "age"])
            .count()
            .reset_index()
        )
        fd4 = fd4[[DEFAULT_ITEM_COL, "age", "t_dat"]].rename(
            {"t_dat": "pop_25_40"}, axis=1
        )
        fd4["pop_25_40"] = (fd4["pop_25_40"] - fd4["pop_25_40"].min()) / (
            fd4["pop_25_40"].max() - fd4["pop_25_40"].min()
        )

        fd5 = (
            fd2[(fd2["age"] > 40) & (fd2["age"] <= 60)]
            .groupby([DEFAULT_ITEM_COL, "age"])
            .count()
            .reset_index()
        )
        fd5 = fd5[[DEFAULT_ITEM_COL, "age", "t_dat"]].rename(
            {"t_dat": "pop_40_60"}, axis=1
        )
        fd5["pop_40_60"] = (fd5["pop_40_60"] - fd5["pop_40_60"].min()) / (
            fd5["pop_40_60"].max() - fd5["pop_40_60"].min()
        )

        fd6 = (
            fd2[fd2["age"] > 60]
            .groupby([DEFAULT_ITEM_COL, "age"])
            .count()
            .reset_index()
        )
        fd6 = fd6[[DEFAULT_ITEM_COL, "age", "t_dat"]].rename(
            {"t_dat": "pop_60"}, axis=1
        )
        fd6["pop_60"] = (fd6["pop_60"] - fd6["pop_60"].min()) / (
            fd6["pop_60"].max() - fd6["pop_60"].min()
        )

        articles = self.dataset.get_articles_df()
        articles = articles[DEFAULT_ITEM_COL]
        articles = pd.merge(fd2, articles, on=DEFAULT_ITEM_COL, how="left")
        articles = articles[
            articles.duplicated(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
        ]

        feature = pd.merge(articles, fd3, on=[DEFAULT_ITEM_COL, "age"], how="left")
        feature = pd.merge(feature, fd4, on=[DEFAULT_ITEM_COL, "age"], how="left")
        feature = pd.merge(feature, fd5, on=[DEFAULT_ITEM_COL, "age"], how="left")
        feature = pd.merge(feature, fd6, on=[DEFAULT_ITEM_COL, "age"], how="left")

        feature = feature[
            [DEFAULT_ITEM_COL, "pop_25", "pop_25_40", "pop_40_60", "pop_60"]
        ].drop_duplicates()

        item_df = self._get_keys_df()
        feature = pd.merge(item_df, feature, on=DEFAULT_ITEM_COL, how="left")
        print(feature)
        return feature
