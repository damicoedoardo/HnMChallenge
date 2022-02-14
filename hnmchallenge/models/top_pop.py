import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.recommender_interface import AbstractRecommender


class TopPop(AbstractRecommender):
    """Top Pop"""

    def __init__(self, dataset: Dataset, time_frame: int = 1):
        """Top Popular items

        Args:
            time_frame (int): last N months to use to compute popularity
        """
        super().__init__(dataset)
        # compute item popularity and cache it
        dr = DataReader()
        full_data = dr.get_full_data()
        # TODO:add timeframe on range of time selected
        # compute popularity
        full_data["pop"] = (
            full_data[full_data["t_dat"] > "2020-08-31"]
            .groupby(DEFAULT_ITEM_COL)[DEFAULT_USER_COL]
            .transform("count")
        )
        item_pop = full_data[[DEFAULT_ITEM_COL, "pop"]].drop_duplicates().fillna(0)
        # cast score to float32 to save memory
        self.item_pop = item_pop.astype({"pop": np.float32})

    def predict(self, interactions: pd.DataFrame) -> pd.DataFrame:
        batch_user_ids = (
            interactions[[DEFAULT_USER_COL]].drop_duplicates().values.squeeze()
        )
        scores = pd.DataFrame(
            np.tile(self.item_pop["pop"].values, (len(batch_user_ids), 1)),
            index=batch_user_ids,
        )
        return scores
