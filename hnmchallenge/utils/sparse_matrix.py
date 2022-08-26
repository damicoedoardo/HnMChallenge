import datetime
import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
from hnmchallenge.constant import *
from hnmchallenge.utils.decorator import timing
from hnmchallenge.utils.logger import set_color
from hnmchallenge.utils.pandas_utils import remap_column_consecutive

logger = logging.getLogger(__name__)


def adjacency_from_interactions(
    interactions: pd.DataFrame, users_num: int, items_num: int
) -> sps.csr_matrix:

    """Return bipartite graph adjacency matrix from interactions

    Args:
        interactions (pd.DataFrame): interactions data
        users_num (int): number of users, used to initialise the shape of the sparse matrix,
        items_num (int): number of items, used to initialise the shape of the sparse matrix

    Returns:
        adjacency (sps.csr_matrix): users interactions in csr sparse format
    """
    for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
        assert c in interactions.columns, f"column {c} not present in train_data"

    user_data = interactions[DEFAULT_USER_COL].values
    item_data = interactions[DEFAULT_ITEM_COL].values + users_num

    row_data = np.concatenate((user_data, item_data), axis=0)
    col_data = np.concatenate((item_data, user_data), axis=0)

    data = np.ones(len(row_data))

    adjacency = sps.csr_matrix(
        (data, (row_data, col_data)),
        shape=(users_num + items_num, items_num + users_num),
    )
    return adjacency


def interactions_to_sparse_matrix(
    interactions: pd.DataFrame,
    users_num: Union[int, None] = None,
    items_num: Union[int, None] = None,
    time_weight: bool = False,
) -> sps.coo_matrix:
    """Convert interactions df into a sparse matrix

    Args:
        interactions (pd.DataFrame): interactions data
        users_num (int): number of users, used to initialise the shape of the sparse matrix
            if None user ids are remapped to consecutive
        items_num (int): number of items, used to initialise the shape of the sparse matrix
            if None item ids are remapped to consecutive

    Returns:
        user_item_matrix (sps.coo_matrix): users interactions in csr sparse format

    """
    for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
        assert c in interactions.columns, f"column {c} not present in train_data"

    row_num = users_num
    col_num = items_num

    user_ids_mapping_dict = None
    item_ids_mapping_dict = None

    if users_num is None:
        interactions, user_ids_mapping_dict = remap_column_consecutive(
            interactions, DEFAULT_USER_COL
        )
        logger.debug(
            set_color("users_num is None, remap user ids to consecutive", "red")
        )
        row_num = len(user_ids_mapping_dict.keys())

    if items_num is None:
        interactions, item_ids_mapping_dict = remap_column_consecutive(
            interactions, DEFAULT_ITEM_COL
        )
        logger.debug(
            set_color("items_num is None, remap item ids to consecutive", "red")
        )
        col_num = len(item_ids_mapping_dict.keys())

    row_data = interactions[DEFAULT_USER_COL].values
    col_data = interactions[DEFAULT_ITEM_COL].values

    if time_weight:
        logger.debug(set_color("Applying time weight on user-item interactions", "red"))

        # interactions["last_buy"] = interactions.groupby(DEFAULT_USER_COL)[
        #     "t_dat"
        # ].transform(max)
        # interactions["first_buy"] = interactions.groupby(DEFAULT_USER_COL)[
        #     "t_dat"
        # ].transform(min)

        # interactions["time_score"] = (
        #     (interactions["t_dat"] - interactions["first_buy"]).apply(lambda x: x.days)
        #     + 1
        # ) / (
        #     (interactions["last_buy"] - interactions["first_buy"]).apply(
        #         lambda x: x.days
        #     )
        #     + 1
        # )

        # # # interactions["time_score"] = interactions["time_score"].fillna(1)
        # interactions["time_score"] = 1 / (1.01 - interactions["time_score"])

        # print(interactions["time_score"])

        # interactions["time_score"] = (
        #     (interactions["t_dat"] - interactions["first_buy"]).apply(lambda x: x.days)
        #     + 1
        # ) / (
        #     (interactions["last_buy"] - interactions["first_buy"]).apply(
        #         lambda x: x.days
        #     )
        #     + 1
        # )

        # min_dat = interactions["t_dat"].min()
        # max_dat = interactions["t_dat"].max()
        # interactions["time_score"] = (interactions["t_dat"] - min_dat) / (
        #     max_dat - min_dat
        # )
        # ) ** 50

        interactions["time_score"] = interactions["t_dat"].apply(
            lambda x: 1 / ((datetime.datetime(2020, 9, 23) - x).days)
        )
        # print(interactions["time_score"])

        data = interactions["time_score"].values
    else:
        data = np.ones(len(row_data))

    user_item_matrix = sps.coo_matrix(
        (data, (row_data, col_data)), shape=(row_num, col_num), dtype="float32"
    )
    return user_item_matrix, user_ids_mapping_dict, item_ids_mapping_dict


def get_top_k(
    scores: np.array,
    top_k: int,
    sort_top_k: bool = True,
    ground_truth: list = None,
):
    """Extract top K element from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (np.array): score matrix (users x items).
        top_k (int): number of top items to recommend.
        sort_top_k (bool): flag to sort top k results.

    Returns:
        np.array, np.array: indices into score matrix for each users top items, scores corresponding to top items.
    """

    # TODO: Maybe do that in multicore
    logger.debug(set_color(f"Sort_top_k:{sort_top_k}", "cyan"))
    # ensure we're working with a dense ndarray
    if isinstance(scores, sps.spmatrix):
        logger.warning(set_color("Scores are in a sparse format, densify them", "red"))
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            set_color(
                "Number of items is less than top_k, limiting top_k to number of items",
                "red",
            )
        )
    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]

    # if groundtruth
    u_gt_filtered, i_gt_filtered, gt_scores_filtered = None, None, None
    if ground_truth is not None:
        u, i = zip(*ground_truth)
        gt_scores = scores[u, i]
        u_gt_filtered, i_gt_filtered, gt_scores_filtered = (
            np.array(u),
            np.array(i),
            np.array(gt_scores),
        )

        # # remove the one hitted
        # u_top_scores = np.repeat(test_user_idx, top_k).reshape(len(test_user_idx), -1)

        # recommended_dict = dict(
        #     zip(zip(u_top_scores, top_items), np.ones(len(u_top_scores)))
        # )

        # u_gt_filtered, i_gt_filtered, gt_scores_filtered = [], [], []

        # gt_list = list(zip(u, i))
        # for tup, s in zip(gt_list, gt_scores):
        #     if tup not in recommended_dict:
        #         u_gt_filtered.append(tup[0])
        #         i_gt_filtered.append(tup[1])
        #         gt_scores_filtered.append(s)

    return (
        np.array(top_items),
        np.array(top_scores),
        u_gt_filtered,
        i_gt_filtered,
        gt_scores_filtered,
    )


def truncate_top_k(x, k):
    """Keep top_k highest values elements for each row of a numpy array

    Args:
        x (np.Array): numpy array
        k (int): number of elements to keep for each row

    Returns:
        np.Array: processed array
    """
    s = x.shape
    ind = np.argpartition(x, -k, axis=1)[:, :-k]
    rows = np.arange(s[0])[:, None]
    x[rows, ind] = 0
    return x
