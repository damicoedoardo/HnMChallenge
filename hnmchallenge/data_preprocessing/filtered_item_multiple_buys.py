import numpy as np
import pandas as pd
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL, RANDOM_SEED
from hnmchallenge.data_reader import DataReader

if __name__ == "__main__":
    dr = DataReader()
    fd = dr.get_filtered_full_data()
    duplicated_rows = fd[fd.duplicated(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])]
    count_mb = duplicated_rows.groupby(DEFAULT_ITEM_COL).count()
    count_mb = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
        columns={"t_dat": "count"}
    )
    count_mb.to_feather(
        dr.get_preprocessed_data_path() / "filtered_multiple_buy.feather"
    )
