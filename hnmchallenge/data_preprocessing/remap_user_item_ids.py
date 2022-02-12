import pickle
from black import main
import pandas as pd
from hnmchallenge.data_reader import DataReader
import numpy as np
from dotenv import load_dotenv
import os


def transaction_to_consecutive_ids() -> None:
    """
    Preprocess transaction df mapping user and items to consecutive ids,
    it also create and save the mapping dictionaries
    """
    dr = DataReader()
    dr.ensure_dirs()

    tr = dr.get_transactions()
    # mapping user ids
    unique_user_ids = tr["customer_id"].unique()
    mapped_ids = np.arange(len(unique_user_ids))
    raw_new_user_ids_dict = dict(zip(unique_user_ids, mapped_ids))
    new_raw_user_ids_dict = {v: k for k, v in raw_new_user_ids_dict.items()}
    tr["customer_id"] = tr["customer_id"].map(raw_new_user_ids_dict.get)
    # mapping item ids
    unique_item_ids = tr["article_id"].unique()
    mapped_ids = np.arange(len(unique_item_ids))
    raw_new_item_ids_dict = dict(zip(unique_item_ids, mapped_ids))
    new_raw_item_ids_dict = {v: k for k, v in raw_new_item_ids_dict.items()}
    tr["article_id"] = tr["article_id"].map(raw_new_item_ids_dict.get)

    df_name = "transactions.feather"
    # save preprocessed df
    tr.to_feather(dr.get_preprocessed_data_path() / df_name)
    # save mapping dictionaries
    dict_dp = dr.get_mapping_dict_path()
    # users
    with open(dict_dp / "raw_new_user_ids_dict.pkl", "wb+") as f:
        pickle.dump(raw_new_user_ids_dict, f)
    with open(dict_dp / "new_raw_user_ids_dict.pkl", "wb+") as f:
        pickle.dump(new_raw_user_ids_dict, f)
    # items
    with open(dict_dp / "raw_new_item_ids_dict.pkl", "wb+") as f:
        pickle.dump(raw_new_item_ids_dict, f)
    with open(dict_dp / "new_raw_item_ids_dict.pkl", "wb+") as f:
        pickle.dump(new_raw_item_ids_dict, f)


if __name__ == "__main__":
    transaction_to_consecutive_ids()
