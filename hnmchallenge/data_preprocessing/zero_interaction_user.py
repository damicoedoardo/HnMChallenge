from hnmchallenge.data_reader import DataReader
from hnmchallenge.constant import *
import numpy as np
import pandas as pd


def save_zero_interaction_users() -> None:
    """Save zero interaction users in feather format"""
    dr = DataReader()
    ss = dr.get_sample_submission()
    target_user = ss[[DEFAULT_USER_COL]]
    full_raw_data = dr.get_transactions()
    unique_train_user = full_raw_data[DEFAULT_USER_COL].unique()
    zero_int_users_df = target_user[
        ~target_user["customer_id"].isin(unique_train_user)
    ].reset_index(drop=True)
    zero_int_users_df.to_feather(
        dr.get_preprocessed_data_path() / "zero_int_users.feather"
    )


if __name__ == "__main__":
    save_zero_interaction_users()
