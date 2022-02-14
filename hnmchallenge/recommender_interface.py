import logging
import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm

from hnmchallenge.constant import *
from hnmchallenge.dataset import Dataset
from hnmchallenge.utils.logger import set_color
from hnmchallenge.utils.sparse_matrix import get_top_k

logger = logging.getLogger(__name__)


class AbstractRecommender(ABC):
    """Interface for recommender system algorithms"""

    name = "Abstract Recommender"

    def __init__(self, dataset: Dataset):
        self.train_data = dataset.get_train_df()
        self.dataset = dataset

    @abstractmethod
    def predict(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Compute items scores for each user inside interactions
        Args:
            interactions (pd.DataFrame): user interactions
        Returns:
            pd.DataFrame: items scores for each user
        """
        pass

    @staticmethod
    def remove_seen_items(
        scores: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Methods to set scores of items used at training time to `-np.inf`

        Args:
            scores (pd.DataFrame): items scores for each user, indexed by user id
            interactions (pd.DataFrane): interactions of the users for which retrieve predictions

        Returns:
            pd.DataFrame: dataframe of scores for each user indexed by user id
        """

        logger.info(set_color(f"Removing seen items", "cyan"))
        user_list = interactions[DEFAULT_USER_COL].values
        item_list = interactions[DEFAULT_ITEM_COL].values

        scores_array = scores.values

        user_index = scores.index.values
        arange = np.arange(len(user_index))
        mapping_dict = dict(zip(user_index, arange))
        user_list_mapped = np.array([mapping_dict.get(u) for u in user_list])

        scores_array[user_list_mapped, item_list] = -np.inf
        scores = pd.DataFrame(scores_array, index=user_index)

        return scores

    def recommend(
        self,
        cutoff: int = 12,
        interactions: pd.DataFrame = None,
        batch_size: int = -1,
    ) -> pd.DataFrame:
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
                If None, predict for the whole users in the training set
            batch_size (int): size of user batch to retrieve recommendations for,
                If -1 no batching procedure is done

        Returns:
            pd.DataFrame: DataFrame with predictions for users
        """
        # if interactions is None we are predicting for the whole users in the train dataset
        if interactions is None:
            interactions = self.train_data

        logger.info(set_color(f"Recommending items", "cyan"))

        user_ids = interactions[DEFAULT_USER_COL].unique()
        # if  batch_size == -1 we are not batching the recommendation process
        num_batches = 1 if batch_size == -1 else math.ceil(len(user_ids) / batch_size)
        user_batches = np.array_split(user_ids, num_batches)

        recs_dfs_list = []
        interactions.set_index([DEFAULT_USER_COL], inplace=True)
        for u_batch in tqdm(user_batches):
            int = interactions[interactions.index.isin(u_batch)]

            # compute scores
            scores = self.predict(int.reset_index())

            # set the score of the items used during the training to -inf
            scores_df = AbstractRecommender.remove_seen_items(scores, int.reset_index())

            array_scores = scores_df.to_numpy()
            user_ids = scores_df.index.values

            # TODO: we can use GPU here (tensorflow ?)
            items, scores = get_top_k(
                scores=array_scores, top_k=cutoff, sort_top_k=True
            )
            # create user array to match shape of retrievied items
            users = np.repeat(user_ids, cutoff).reshape(len(user_ids), -1)

            recs_df = pd.DataFrame(
                zip(users.flatten(), items.flatten(), scores.flatten()),
                columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL],
            )
            recs_dfs_list.append(recs_df)
        # move customer id from index to be a column
        interactions.reset_index(inplace=True)

        # concat all the batch recommendations dfs
        recommendation_df = pd.concat(recs_dfs_list, axis=0)
        # add item rank
        recommendation_df["rank"] = np.tile(np.arange(1, cutoff + 1), len(user_ids))
        return recommendation_df
