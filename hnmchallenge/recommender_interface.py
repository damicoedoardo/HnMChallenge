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
import similaripy
import torch
from p_tqdm import p_imap, p_map
from pathos.pools import ProcessPool
from tqdm import tqdm

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.utils.decorator import timing
from hnmchallenge.utils.logger import set_color
from hnmchallenge.utils.sparse_matrix import get_top_k, interactions_to_sparse_matrix

logger = logging.getLogger(__name__)


class AbstractRecommender(ABC):
    """Interface for recommender system algorithms"""

    name = "Abstract Recommender"

    def __init__(self, dataset):
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

        logger.debug(set_color(f"Removing seen items", "cyan"))

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

    @staticmethod
    def filter_on_candidate(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        # Filter computed scores on a set of candidates items setting the scores of all the other items to -np.inf
        print("Filtering scores on candidates...")
        # creating mask for candidate items
        mask_array = np.ones(scores.shape[1], dtype=bool)
        mask_array[candidates] = False
        scores[:, mask_array] = -np.inf
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
        logger.debug(set_color(f"Recommending items MONOCORE", "cyan"))

        unique_user_ids = interactions[DEFAULT_USER_COL].unique()
        logger.debug(set_color(f"Predicting for: {len(unique_user_ids)} users", "cyan"))
        # if  batch_size == -1 we are not batching the recommendation process
        num_batches = (
            1 if batch_size == -1 else math.ceil(len(unique_user_ids) / batch_size)
        )
        logger.debug(set_color(f"num batches: {num_batches}", "cyan"))
        user_batches = np.array_split(unique_user_ids, num_batches)

        # MONO-CORE VERSION
        recs_dfs_list = []
        for user_batch in tqdm(user_batches):
            interactions_slice = interactions[
                interactions[DEFAULT_USER_COL].isin(user_batch)
            ]
            logger.debug(set_color(f"getting predictions...", "cyan"))
            scores = self.predict(interactions_slice)
            logger.debug(set_color(f"done...", "cyan"))
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
        filter_on_candidates: bool = False,
        insert_gt: bool = False,
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
        logger.debug(set_color(f"Recommending items MULTICORE", "cyan"))

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

        def _rec(
            interactions_df, white_list_mb_item=None, candidates=None, ground_truth=None
        ):
            scores = self.predict(interactions_df)
            # set the score of the items used during the training to -inf
            if remove_seen:
                scores = AbstractRecommender.remove_seen_items(
                    scores, interactions_df, white_list_mb_item
                )
            array_scores = scores.to_numpy()

            # filter the scores on a subset of candidates items
            if candidates is not None:
                array_scores = AbstractRecommender.filter_on_candidate(
                    array_scores, candidates
                )

            user_ids = scores.index.values

            final_gt = None
            if ground_truth is not None:
                print("Including ground_truth...")
                # 1) filter on the user we have
                # 2) map the user id to the proper one
                filtered_gt = ground_truth[
                    ground_truth[DEFAULT_USER_COL].isin(user_ids)
                ]

                md = dict(zip(user_ids, np.arange(len(user_ids))))
                inverse_md = {v: k for k, v in md.items()}

                filtered_gt[DEFAULT_USER_COL] = filtered_gt[DEFAULT_USER_COL].apply(
                    lambda x: md.get(x)
                )

                user = filtered_gt[DEFAULT_USER_COL].values
                item = filtered_gt[DEFAULT_ITEM_COL].values
                final_gt = list(zip(user, item))

            items, scores, u_gt_filtered, i_gt_filtered, gt_scores_filtered = get_top_k(
                scores=array_scores,
                top_k=cutoff,
                sort_top_k=True,
                ground_truth=final_gt,
            )

            # create user array to match shape of retrievied items
            users = np.repeat(user_ids, cutoff).reshape(len(user_ids), -1)

            recs_df = pd.DataFrame(
                zip(users.flatten(), items.flatten(), scores.flatten()),
                columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL],
            )

            # for gt
            if u_gt_filtered is not None:
                u_gt_filtered_mapped = np.array(
                    list(map(lambda x: inverse_md.get(x), u_gt_filtered))
                )
                recs_gt = pd.DataFrame(
                    zip(u_gt_filtered_mapped, i_gt_filtered, gt_scores_filtered),
                    columns=[
                        DEFAULT_USER_COL,
                        DEFAULT_ITEM_COL,
                        DEFAULT_PREDICTION_COL,
                    ],
                )
                # overwrite recs_df
                recs_df = pd.concat([recs_df, recs_gt], axis=0)
                # drop duplicates created by hits
                recs_df = recs_df.drop_duplicates()

                # but has to be sorted!
                recs_df = recs_df.sort_values(
                    [DEFAULT_USER_COL, DEFAULT_PREDICTION_COL], ascending=[True, False]
                )

                recs_per_user = recs_df.groupby(DEFAULT_USER_COL).size().values
                ensemble_rank = np.concatenate(
                    list(map(lambda x: np.arange(1, x + 1), recs_per_user))
                )
                recs_df["rank"] = ensemble_rank
            else:
                # add item rank when no gt
                recs_df["rank"] = np.tile(np.arange(1, cutoff + 1), len(user_ids))
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
            elif (filter_on_candidates) and (not insert_gt):
                # pass the candidates
                # load candidates
                candidate_items = self.dataset.get_candidate_items()
                reps_candidate_items = np.repeat(
                    np.array(candidate_items)[np.newaxis, :], len(train_dfs), axis=0
                )

                # Creating an array of None for the white list item arg on _rec() function
                reps_white_list_mb_item = [None] * len(train_dfs)

                recs_dfs_list = p_map(
                    _rec,
                    train_dfs,
                    reps_white_list_mb_item,
                    reps_candidate_items,
                    num_cpus=num_cpus,
                )
            elif filter_on_candidates and insert_gt:
                # pass the candidates
                # load candidates
                candidate_items = self.dataset.get_candidate_items()
                reps_candidate_items = np.repeat(
                    np.array(candidate_items)[np.newaxis, :], len(train_dfs), axis=0
                )

                # Creating an array of None for the white list item arg on _rec() function
                reps_white_list_mb_item = [None] * len(train_dfs)

                gt = self.dataset.get_holdout()[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
                reps_gt = [gt.copy() for _ in range(len(train_dfs))]

                recs_dfs_list = p_map(
                    _rec,
                    train_dfs,
                    reps_white_list_mb_item,
                    reps_candidate_items,
                    reps_gt,
                    num_cpus=num_cpus,
                )

            else:
                recs_dfs_list = p_map(_rec, train_dfs, num_cpus=num_cpus)

        # concat all the batch recommendations dfs
        recommendation_df = pd.concat(recs_dfs_list, axis=0)

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
            logger.debug(set_color("Predicting using time_weight importance...", "red"))
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interactions,
            items_num=self.dataset._ARTICLES_NUM,
            users_num=None,
            time_weight=self.time_weight,
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        if not sps.issparse(self.similarity_matrix):
            logger.debug(set_color(f"DENSE Item Similarity MUL...", "cyan"))
            scores = sparse_interaction @ self.similarity_matrix
        else:
            logger.debug(set_color(f"SPARSE Item Similarity MUL...", "cyan"))
            scores = sparse_interaction @ self.similarity_matrix
            scores = scores.toarray()

        scores_df = pd.DataFrame(scores, index=list(user_mapping_dict.keys()))
        return scores_df


class UserSimilarityRecommender(AbstractRecommender, ABC):
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

    def predict(self, interactions: pd.DataFrame, cutoff: int, remove_seen: bool):
        assert (
            self.similarity_matrix is not None
        ), "Similarity matrix is not computed, call compute_similarity_matrix()"
        if self.time_weight:
            logger.debug(set_color("Predicting using time_weight importance...", "red"))
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interactions,
            items_num=self.dataset._ARTICLES_NUM,
            users_num=None,
            time_weight=self.time_weight,
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        if not sps.issparse(self.similarity_matrix):
            raise NotImplementedError(
                "user similarity can only be used with sparse similarity matrices!"
            )
        else:
            logger.debug(set_color(f"SPARSE Item Similarity MUL...", "cyan"))

            print(self.similarity_matrix.shape)
            print(sparse_interaction.shape)
            print(cutoff)

            filter_cols = sparse_interaction if remove_seen else None
            scores = similaripy.dot_product(
                self.similarity_matrix.T,
                sparse_interaction,
                k=cutoff,
                # filter_cols=filter_cols,
            )
            # scores = self.similarity_matrix.dot(sparse_interaction)
            scores = scores.toarray()
            print(scores)

        scores_df = pd.DataFrame(scores, index=list(user_mapping_dict.keys()))
        return scores_df


class RepresentationBasedRecommender(AbstractRecommender, ABC):
    """Representation based recommender interface"""

    def __init__(self, dataset):
        super().__init__(dataset)

    @abstractmethod
    def compute_representations(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Compute users and items representations

        Args:
            interactions (pd.Dataframe): interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame, pd.DataFrame: user representations, item representations
        """
        pass

    def predict(self, interactions):
        # TODO CHECK THAT

        # we user the dor product between user and item embeddings to predict the user preference scores
        users_repr_df, items_repr_df = self.compute_representations(interactions)

        assert isinstance(users_repr_df, pd.DataFrame) and isinstance(
            items_repr_df, pd.DataFrame
        ), "Representations have to be stored inside pd.DataFrane objects!\n user: {}, item: {}".format(
            type(users_repr_df), type(items_repr_df)
        )
        assert (
            users_repr_df.shape[1] == items_repr_df.shape[1]
        ), "Users and Items representations have not the same shape!\n user: {}, item: {}".format(
            users_repr_df.shape[1], items_repr_df.shape[1]
        )

        # sort items representations
        items_repr_df.sort_index(inplace=True)

        # compute the scores as dot product between users and items representations
        arr_scores = users_repr_df.to_numpy().dot(items_repr_df.to_numpy().T)
        scores = pd.DataFrame(arr_scores, index=users_repr_df.index)
        return scores
