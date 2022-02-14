import logging
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps
from hnmchallenge.constant import *
from hnmchallenge.utils.logger import set_color

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
    interactions: pd.DataFrame, users_num: int, items_num: int
) -> sps.coo_matrix:
    """Convert interactions df into a sparse matrix

    Args:
        interactions (pd.DataFrame): interactions data
        users_num (int): number of users, used to initialise the shape of the sparse matrix
        items_num (int): number of items, used to initialise the shape of the sparse matrix

    Returns:
        user_item_matrix (sps.coo_matrix): users interactions in csr sparse format

    """
    for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
        assert c in interactions.columns, f"column {c} not present in train_data"

    row_num = users_num
    col_num = items_num

    row_data = interactions[DEFAULT_USER_COL].values
    col_data = interactions[DEFAULT_ITEM_COL].values
    data = np.ones(len(row_data))

    user_item_matrix = sps.coo_matrix(
        (data, (row_data, col_data)), shape=(row_num, col_num)
    )
    return user_item_matrix


def get_top_k(
    scores: np.array, top_k: int, sort_top_k: bool = True
) -> Tuple[np.array, np.array]:
    """Extract top K element from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (np.array): score matrix (users x items).
        top_k (int): number of top items to recommend.
        sort_top_k (bool): flag to sort top k results.

    Returns:
        np.array, np.array: indices into score matrix for each users top items, scores corresponding to top items.
    """
    # TODO: Maybe do that in multicore

    logger.info(set_color(f"Sort_top_k:{sort_top_k}", "cyan"))
    # ensure we're working with a dense ndarray
    if isinstance(scores, sps.spmatrix):
        logger.warning(
            set_color("Scores are in a sparse format, densify them", "white")
        )
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            set_color(
                "Number of items is less than top_k, limiting top_k to number of items",
                "white",
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

    return np.array(top_items), np.array(top_scores)
