from black import main
import pandas as pd
from hnmchallenge.data_reader import DataReader
import numpy as np

if __name__ == "__main__":
    dr = DataReader()
    full_data = dr.get_full_data()
    # create val and test being a week of data each
    # timestamp on train is max: 2020-09-22 min: 2018-09-20

    # I have to drop the indices to save in Feather format
    test_df = full_data[full_data["t_dat"] > "2020-09-14"].reset_index(drop=True)

    val_df = full_data[
        (full_data["t_dat"] > "2020-09-7") & (full_data["t_dat"] < "2020-09-15")
    ].reset_index(drop=True)

    train_df = full_data[(full_data["t_dat"] < "2020-09-8")].reset_index(drop=True)
    # create a small train taking a month of data (july data)
    train_small_df = full_data[
        (full_data["t_dat"] > "2020-07-30") & (full_data["t_dat"] < "2020-09-8")
    ].reset_index(drop=True)

    # save dfs
    train_small_df.to_feather(
        dr.get_preprocessed_data_path() / "small_train_df.feather"
    )
    train_df.to_feather(dr.get_preprocessed_data_path() / "train_df.feather")
    val_df.to_feather(dr.get_preprocessed_data_path() / "val_df.feather")
    test_df.to_feather(dr.get_preprocessed_data_path() / "test_df.feather")
