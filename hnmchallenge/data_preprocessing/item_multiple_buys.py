import numpy as np
import pandas as pd
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL, RANDOM_SEED
from hnmchallenge.data_reader import DataReader

if __name__ == "__main__":
    dr = DataReader()
    full_data = dr.get_full_data()
    count = full_data.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).count()
    # check which items have been bought multiple times from the same user
    multiple_buy = count[count["price"] > 1]
    final_count = (
        multiple_buy.reset_index()
        .groupby(DEFAULT_ITEM_COL)
        .sum()[["price"]]
        .reset_index()
        .rename(columns={"price": "count"})
    )
    final_count.to_feather(dr.get_preprocessed_data_path() / "multiple_buy.feather")
