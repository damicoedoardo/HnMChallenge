import numpy as np
import pandas as pd
from hnmchallenge.constant import DEFAULT_USER_COL, RANDOM_SEED
from hnmchallenge.data_reader import DataReader

if __name__ == "__main__":
    dr = DataReader()
    full_data = dr.get_filtered_full_data()
    # create val and test being a week of data each
    # timestamp on train is max: 2020-09-22 min: 2018-09-20

    # I have to drop the indices to save in Feather format
    last_week_data = full_data[full_data["t_dat"] > "2020-09-14"].reset_index(drop=True)
    # split the users in two and create val and test df
    unique_users = last_week_data[DEFAULT_USER_COL].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_users)
    val_users, test_users = np.array_split(unique_users, 2)

    val_data = last_week_data[
        last_week_data[DEFAULT_USER_COL].isin(val_users)
    ].reset_index(drop=True)
    test_data = last_week_data[
        last_week_data[DEFAULT_USER_COL].isin(test_users)
    ].reset_index(drop=True)

    print(f"validation data len: {len(val_data)}")
    print(f"test data len: {len(test_data)}")

    train_data = full_data[full_data["t_dat"] <= "2020-09-14"].reset_index(drop=True)
    print(f"train data len: {len(train_data)}")

    # save splits
    train_data.to_feather(dr.get_preprocessed_data_path() / "filtered_train_df.feather")
    val_data.to_feather(dr.get_preprocessed_data_path() / "filtered_val_df.feather")
    test_data.to_feather(dr.get_preprocessed_data_path() / "filtered_test_df.feather")

    # get a subset of 50k users
    USERS_SUBSET_NUM = 50_000
    np.random.shuffle(unique_users)
    user_sub = unique_users[:USERS_SUBSET_NUM]
    train_small = train_data[train_data[DEFAULT_USER_COL].isin(user_sub)].reset_index(
        drop=True
    )
    print(f"train small len: {len(train_small)}")
    # save train small
    train_small.to_feather(
        dr.get_preprocessed_data_path() / "filtered_train_small_df.feather"
    )
