import pandas as pd


def adjacency_from_interactions(
    interactions: pd.DataFrame, users_num=None, items_num=None
):
    """Return adjacency from interactions

    Args:
        interactions (pd.DataFrame): interactions data
        users_num (int): number of users, used to initialise the shape of the sparse matrix,
            if None user ids are remapped to consecutive
        items_num (int): number of items, used to initialise the shape of the sparse matrix
            if None user ids are remapped to consecutive

    Returns:
        sp_m (sps.csr_matrix): users interactions in csr sparse format
        user_ids_mapping_dict (dict): dictionary mapping user ids to the rows of the sparse matrix
        user_ids_mapping_dict (dict): dictionary mapping item ids to the cols of the sparse matirx
    """
    for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
        assert c in interactions.columns, f"column {c} not present in train_data"

    row_num = users_num
    col_num = items_num

    if users_num is None:
        interactions, user_ids_mapping_dict = pu.remap_column_consecutive(
            interactions, DEFAULT_USER_COL
        )
        logger.warning(
            set_color("users_num is None, remap user ids to consecutive", "white")
        )
        row_num = len(user_ids_mapping_dict.keys())

    if items_num is None:
        interactions, item_ids_mapping_dict = pu.remap_column_consecutive(
            interactions, DEFAULT_ITEM_COL
        )
        logger.warning(
            set_color("items_num is None, remap item ids to consecutive", "white")
        )
        col_num = len(item_ids_mapping_dict.keys())

    user_data = interactions[DEFAULT_USER_COL].values
    item_data = interactions[DEFAULT_ITEM_COL].values + row_num

    row_data = np.concatenate((user_data, item_data), axis=0)
    col_data = np.concatenate((item_data, user_data), axis=0)

    data = np.ones(len(row_data))

    adj = sps.csr_matrix(
        (data, (row_data, col_data)), shape=(row_num + col_num, col_num + row_num)
    )
    return adj
