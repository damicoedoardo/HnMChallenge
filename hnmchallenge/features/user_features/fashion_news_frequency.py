from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from hnmchallenge.dataset import Dataset
from hnmchallenge.features.feature_interfaces import UserFeature
from hnmchallenge.stratified_dataset import StratifiedDataset


class FashionNewsFrequency(UserFeature):
    FEATURE_NAME = "fashion_news_frequency"

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        user_df = self.dr.get_full_customers()
        fnf = pd.get_dummies(user_df["fashion_news_frequency"])
        user = user_df[DEFAULT_USER_COL].to_frame()
        feature = user.join(fnf)
        return feature


if __name__ == "__main__":
    dataset = Dataset()
    feature = FashionNewsFrequency(dataset, kind="full")
    feature.save_feature()
