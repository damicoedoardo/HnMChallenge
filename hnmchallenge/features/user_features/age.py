from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.dataset import Dataset
from hnmchallenge.features.feature_interfaces import UserFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class Age(UserFeature):
    FEATURE_NAME = "age"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        feature = self.dr.get_full_customers()[[DEFAULT_USER_COL, "age"]]
        return feature


if __name__ == "__main__":
    dataset = Dataset()
    feature = Age(dataset, kind="full")
    feature.save_feature()
