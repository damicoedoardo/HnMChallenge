import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
import datetime

if __name__ == "__main__":
    dr = DataReader()
    fd = dr.get_filtered_full_data()
    fd["tdiff"]=fd['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,23) - x).days)
    fd["max_price"] = fd.groupby(DEFAULT_ITEM_COL)["price"].transform("max")
    fd["sale_factor"] = (1 - (fd["price"] / fd["max_price"]))
    fd["last_buy"] = fd.groupby(DEFAULT_USER_COL)["t_dat"].transform(max)
    fd["first_buy"] = fd.groupby(DEFAULT_USER_COL)["t_dat"].transform(min)
    fd["time_score"] = ((fd["t_dat"] - fd["first_buy"])/ (fd["last_buy"] - fd["first_buy"])) 
    fd=fd.drop(["last_buy", "first_buy"], axis =1)
    fd.to_feather(
        dr.get_preprocessed_data_path() / "filtered_feature_dataset.feather"
    )

