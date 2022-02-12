import pandas as pd
from hnmchallenge.constant import *
import numpy as np
import scipy.sparse as sps
import logging
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

    user_ids_mapping_dict = None
    item_ids_mapping_dict = None

    row_data = interactions[DEFAULT_USER_COL].values
    col_data = interactions[DEFAULT_ITEM_COL].values
    data = np.ones(len(row_data))

    user_item_matrix = sps.coo_matrix(
        (data, (row_data, col_data)), shape=(row_num, col_num)
    )
    return user_item_matrix
