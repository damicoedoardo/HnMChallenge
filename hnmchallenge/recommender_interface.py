import logging
import math
from abc import ABC, abstractmethod
from textwrap import dedent
from time import time
from tkinter.messagebox import NO
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from p_tqdm import p_imap, p_map
from pathos.pools import ProcessPool
from tqdm import tqdm

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.utils.decorator import timing
from hnmchallenge.utils.logger import set_color
from hnmchallenge.utils.sparse_matrix import get_top_k, interactions_to_sparse_matrix

logger = logging.getLogger(__name__)


class AbstractRecommender(ABC):
    """Interface for recommender system algorithms"""

    name = "Abstract Recommender"

    def __init__(self, dataset: Dataset):
        # self.train_data = dataset.get_train_df()
        self.dataset = dataset
        # to be set
        self.item_multiple_buy = None

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
        scores: pd.DataFrame,
        interactions: pd.DataFrame,
        white_list_mb_item: np.array = None,
    ) -> pd.DataFrame:
        """Methods to set scores of items used at training time to `-np.inf`

        Args:
            scores (pd.DataFrame): items scores for each user, indexed by user id
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
            items_multiple_buy (pd.DataFrame): items that can be recommended multiple times

        Returns:
            pd.DataFrame: dataframe of scores for each user indexed by user id
        """

        logger.info(set_color(f"Removing seen items", "cyan"))

        if white_list_mb_item is not None:
            print("Considering white list items...")
            interactions = interactions[
                ~(interactions[DEFAULT_ITEM_COL].isin(white_list_mb_item))
            ]

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
        interactions: pd.DataFrame,
        cutoff: int = 12,
        remove_seen: bool = True,
        batch_size: int = -1,
    ) -> pd.DataFrame:
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
            batch_size (int): size of user batch to retrieve recommendations for,
                If -1 no batching procedure is done
            remove_seen (bool): remove items that have been bought ones from the prediction

        Returns:
            pd.DataFrame: DataFrame with predictions for users
        """
        # if interactions is None we are predicting for the wh  ole users in the train dataset
        logger.info(set_color(f"Recommending items MONOCORE", "cyan"))

        unique_user_ids = interactions[DEFAULT_USER_COL].unique()
        logger.info(set_color(f"Predicting for: {len(unique_user_ids)} users", "cyan"))
        # if  batch_size == -1 we are not batching the recommendation process
        num_batches = (
            1 if batch_size == -1 else math.ceil(len(unique_user_ids) / batch_size)
        )
        logger.info(set_color(f"num batches: {num_batches}", "cyan"))
        user_batches = np.array_split(unique_user_ids, num_batches)

        # MONO-CORE VERSION
        recs_dfs_list = []
        for user_batch in tqdm(user_batches):
            interactions_slice = interactions[
                interactions[DEFAULT_USER_COL].isin(user_batch)
            ]
            logger.info(set_color(f"getting predictions...", "cyan"))
            scores = self.predict(interactions_slice)
            logger.info(set_color(f"done...", "cyan"))
            # set the score of the items used during the training to -inf
            if remove_seen:
                scores = AbstractRecommender.remove_seen_items(
                    scores, interactions_slice
                )
            array_scores = scores.to_numpy()
            user_ids = scores.index.values
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

        # concat all the batch recommendations dfs
        recommendation_df = pd.concat(recs_dfs_list, axis=0)
        # add item rank
        recommendation_df["rank"] = np.tile(
            np.arange(1, cutoff + 1), len(unique_user_ids)
        )

        return recommendation_df

    def recommend_multicore(
        self,
        interactions: pd.DataFrame,
        cutoff: int = 12,
        remove_seen: bool = True,
        white_list_mb_item: Union[np.array, None] = None,
        batch_size: int = -1,
        num_cpus: int = 5,
    ) -> pd.DataFrame:
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
            batch_size (int): size of user batch to retrieve recommendations for,
                If -1 no batching procedure is done
            num_cpus (int): number of cores to use to parallelise batch recommendations
            remove_seen (bool): remove items that have been bought ones from the prediction

        Returns:
            pd.DataFrame: DataFrame with predictions for users
        """
        # if interactions is None we are predicting for the wh  ole users in the train dataset
        logger.info(set_color(f"Recommending items MULTICORE", "cyan"))

        unique_user_ids = interactions[DEFAULT_USER_COL].unique()
        # if  batch_size == -1 we are not batching the recommendation process
        num_batches = (
            1 if batch_size == -1 else math.ceil(len(unique_user_ids) / batch_size)
        )
        user_batches = np.array_split(unique_user_ids, num_batches)

        # MULTI-CORE VERSION
        train_dfs = [
            interactions[interactions[DEFAULT_USER_COL].isin(u_batch)]
            for u_batch in user_batches
        ]

        def _rec(interactions_df, white_list_mb_item=None):
            scores = self.predict(interactions_df)
            # set the score of the items used during the training to -inf
            if remove_seen:
                scores = AbstractRecommender.remove_seen_items(
                    scores, interactions_df, white_list_mb_item
                )
            array_scores = scores.to_numpy()
            user_ids = scores.index.values
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
            return recs_df

        if batch_size == -1:
            recs_dfs_list = [_rec(train_dfs[0])]
        else:
            # pool = ProcessPool(nodes=num_cpus)
            # results = pool.imap(_rec, train_dfs)
            # recs_dfs_list = list(results)
            if white_list_mb_item is not None:
                reps_white_list_mb_item = np.repeat(
                    np.array(white_list_mb_item)[np.newaxis, :], len(train_dfs), axis=0
                )
                recs_dfs_list = p_map(
                    _rec, train_dfs, reps_white_list_mb_item, num_cpus=num_cpus
                )
            else:
                recs_dfs_list = p_map(_rec, train_dfs, num_cpus=num_cpus)

        # concat all the batch recommendations dfs
        recommendation_df = pd.concat(recs_dfs_list, axis=0)
        # add item rank
        recommendation_df["rank"] = np.tile(
            np.arange(1, cutoff + 1), len(unique_user_ids)
        )

        return recommendation_df


class ItemSimilarityRecommender(AbstractRecommender, ABC):
    """Item similarity matrix recommender interface

    Each recommender extending this class has to implement compute_similarity_matrix() method
    """

    def __init__(self, dataset, time_weight: bool = False):
        super().__init__(dataset=dataset)
        self.time_weight = time_weight
        self.similarity_matrix = None

    @abstractmethod
    def compute_similarity_matrix(self):
        """Compute similarity matrix and assign it to self.similarity_matrix"""
        pass

    def predict(self, interactions):
        assert (
            self.similarity_matrix is not None
        ), "Similarity matrix is not computed, call compute_similarity_matrix()"
        if self.time_weight:
            logger.info(set_color("Predicting using time_weight importance...", "red"))
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interactions,
            items_num=self.dataset._ARTICLES_NUM,
            users_num=None,
            time_weight=self.time_weight,
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        if not sps.issparse(self.similarity_matrix):
            logger.info(set_color(f"DENSE Item Similarity MUL...", "cyan"))
            scores = sparse_interaction @ self.similarity_matrix

            # gpu
            # dense_interactions = sparse_interaction.toarray()

            # # construc torch tensors
            # sim_tensor = torch.from_numpy(self.similarity_matrix)
            # interactions_tensor = torch.from_numpy(dense_interactions)

            # sim_tensor_list = sim_tensor.split(10_000)
            # interactions_tensor_list = interactions_tensor.split(10_000, dim=1)

            # res = []
            # torch.cuda.empty_cache()
            # for sim_tensor, interactions_tensor in tqdm(
            #     zip(sim_tensor_list, interactions_tensor_list)
            # ):
            #     print("a")
            #     part_res = torch.matmul(
            #         interactions_tensor.to("cuda"), sim_tensor.to("cuda")
            #     )
            #     res.append(part_res.cpu().numpy())
            #     torch.cuda.empty_cache()

            # scores_torch = torch.concat(res)

            # # bring scores to cpu and convert to numpy
            # scores = scores_torch.cpu().numpy()
        else:
            logger.info(set_color(f"SPARSE Item Similarity MUL...", "cyan"))
            scores = sparse_interaction @ self.similarity_matrix
            scores = scores.toarray()

        scores_df = pd.DataFrame(scores, index=list(user_mapping_dict.keys()))
        return scores_df
