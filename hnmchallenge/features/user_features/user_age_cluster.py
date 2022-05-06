from unicodedata import name

import numpy as np
import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserFeature


class UserAgeCluster(UserFeature):
    FEATURE_NAME = "user_age_cluster"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        customers = self.dataset.get_customers_df()[[DEFAULT_USER_COL, "age"]]
        customers["user_age_cluster"] = 0
        customers.loc[
            (customers["age"] > 25) & (customers["age"] <= 40), "user_age_cluster"
        ] = 1
        customers.loc[
            (customers["age"] > 40) & (customers["age"] <= 60), "user_age_cluster"
        ] = 2
        customers.loc[
            (customers["age"] > 60) & (customers["age"] <= 80), "user_age_cluster"
        ] = 3
        customers.loc[customers["age"] > 80, "user_age_cluster"] = 4
        feature = customers[[DEFAULT_USER_COL, "user_age_cluster"]]

        return feature
