import numpy as np
import pandas as pd
from hnmchallenge.constant import (DEFAULT_ITEM_COL, DEFAULT_USER_COL,
                                   RANDOM_SEED)
from hnmchallenge.data_reader import DataReader

if __name__ == "__main__":
    dr = DataReader()
    full_data = dr.get_full_data()
    item_last_month = full_data[full_data["t_dat"] >= "2020-09-1"][
        DEFAULT_ITEM_COL
    ].unique()
    with open(dr.get_preprocessed_data_path() / "item_last_month.npy", "wb+") as f:
        np.save(f, item_last_month)
    
