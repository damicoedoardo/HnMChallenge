import logging

import numpy as np
import pandas as pd
from hnmchallenge.utils.logger import set_color

logger = logging.getLogger(__name__)


def remap_column_consecutive(df, column_name):
    """Remap selected column of a given dataframe into consecutive numbers

    Args:
        df (pd.DataFrame): dataframe
        column_name (str, int): name of the column to remap
    """
    assert (
        column_name in df.columns.values
    ), f"Column name: {column_name} not in df.columns: {df.columns.values}"

    copy_df = df.copy()
    unique_data = copy_df[column_name].unique()
    logger.debug(set_color(f"unique {column_name}: {len(unique_data)}", "yellow"))
    data_idxs = np.arange(len(unique_data), dtype=np.int)
    data_idxs_map = dict(zip(unique_data, data_idxs))
    copy_df[column_name] = copy_df[column_name].map(data_idxs_map.get)
    return copy_df, data_idxs_map
