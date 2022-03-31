import os
import pickle

import numpy as np
import pandas as pd
from black import main
from dotenv import load_dotenv
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader


def transaction_to_consecutive_ids() -> None:
    """
    Preprocess transaction df mapping user and items to consecutive ids,
    it also create and save the mapping dictionaries
    """
    dr = DataReader()
    dr.ensure_dirs()

    tr = dr.get_transactions()

    item_last_month = tr[tr["t_dat"] >= "2020-09-1"][DEFAULT_ITEM_COL].unique()
    # filter on that users
    filtered_tr = tr[tr[DEFAULT_ITEM_COL].isin(item_last_month)]

    # mapping user ids
    unique_user_ids = filtered_tr["customer_id"].unique()

    # save the users which have no interactions with the item subset
    ss = dr.get_sample_submission()
    target_user = ss[[DEFAULT_USER_COL]]
    zero_int_users_df = target_user[
        ~target_user["customer_id"].isin(unique_user_ids)
    ].reset_index(drop=True)
    print(f"len user no interactions: {len(zero_int_users_df)}")

    # zero_int_users_df.to_feather(
    #     dr.get_preprocessed_data_path() / "filtered_zero_int_users.feather"
    # )

    print(f"Unique users: {len(unique_user_ids)}")
    mapped_ids = np.arange(len(unique_user_ids))
    raw_new_user_ids_dict = dict(zip(unique_user_ids, mapped_ids))
    new_raw_user_ids_dict = {v: k for k, v in raw_new_user_ids_dict.items()}
    filtered_tr["customer_id"] = filtered_tr["customer_id"].map(
        raw_new_user_ids_dict.get
    )
    # mapping item ids
    unique_item_ids = filtered_tr["article_id"].unique()
    print(f"Unique items: {len(unique_item_ids)}")
    mapped_ids = np.arange(len(unique_item_ids))
    raw_new_item_ids_dict = dict(zip(unique_item_ids, mapped_ids))
    new_raw_item_ids_dict = {v: k for k, v in raw_new_item_ids_dict.items()}
    filtered_tr["article_id"] = filtered_tr["article_id"].map(raw_new_item_ids_dict.get)

    # df_name = "filtered_transactions.feather"
    # # save preprocessed df
    # filtered_tr = filtered_tr.reset_index(drop=True)
    # filtered_tr.to_feather(dr.get_preprocessed_data_path() / df_name)
    # # save mapping dictionaries
    # dict_dp = dr.get_mapping_dict_path()
    # # users
    # with open(dict_dp / "filtered_raw_new_user_ids_dict.pkl", "wb+") as f:
    #     pickle.dump(raw_new_user_ids_dict, f)
    # with open(dict_dp / "filtered_new_raw_user_ids_dict.pkl", "wb+") as f:
    #     pickle.dump(new_raw_user_ids_dict, f)
    # # items
    # with open(dict_dp / "filtered_raw_new_item_ids_dict.pkl", "wb+") as f:
    #     pickle.dump(raw_new_item_ids_dict, f)
    # with open(dict_dp / "filtered_new_raw_item_ids_dict.pkl", "wb+") as f:
    #     pickle.dump(new_raw_item_ids_dict, f)


if __name__ == "__main__":
    transaction_to_consecutive_ids()
