import numpy as np
import pandas as pd
from hnmchallenge.constant import DEFAULT_USER_COL, RANDOM_SEED
from hnmchallenge.data_reader import DataReader

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

    # filter on a subset of the users
    USERS_NUM = 100_000
    # fix numpy random seed
    np.random.seed(RANDOM_SEED)
    customer_ids = full_data["customer_id"].unique()
    # shuffle customer ids
    np.random.shuffle(customer_ids)
    customer_subset_ids = customer_ids[:USERS_NUM]

    # filter train, val and test on the user subset
    train_df = train_df[
        train_df[DEFAULT_USER_COL].isin(customer_subset_ids)
    ].reset_index(drop=True)
    val_df = val_df[val_df[DEFAULT_USER_COL].isin(customer_subset_ids)].reset_index(
        drop=True
    )
    test_df = test_df[test_df[DEFAULT_USER_COL].isin(customer_subset_ids)].reset_index(
        drop=True
    )

    # save the filtered data
    train_df.to_feather(
        dr.get_preprocessed_data_path() / "train_df_user_subset.feather"
    )
    val_df.to_feather(dr.get_preprocessed_data_path() / "val_df_user_subset.feather")
    test_df.to_feather(dr.get_preprocessed_data_path() / "test_df_user_subset.feather")
