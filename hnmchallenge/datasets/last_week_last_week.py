import pickle

import numpy as np
from hnmchallenge.constant import *
from hnmchallenge.dataset_interface import DatasetInterface


class LWLWDataset(DatasetInterface):

    DATASET_NAME = "LWLW_dataset"
    _ARTICLES_NUM = 18_684
    _CUSTOMERS_NUM = 1_105_000

    def __init__(self) -> None:
        super().__init__()

    def create_dataset_description(self) -> str:
        description = """ 
        Items: t_dat > 15/09/2020 \n
        holdout: last_week
        """
        return description

    def remap_user_item_ids(self) -> None:
        tr = self.dr.get_transactions()

        # filter on the items present in the last month
        item_last_week = tr[tr["t_dat"] >= "2020-09-15"][DEFAULT_ITEM_COL].unique()
        tr = tr[tr[DEFAULT_ITEM_COL].isin(item_last_week)]

        print(f"Unique users: {tr[DEFAULT_USER_COL].nunique()}")
        print(f"Unique items: {tr[DEFAULT_ITEM_COL].nunique()}")

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

        # save preprocessed df
        tr.reset_index(drop=True).to_feather(self._FULL_DATA_PATH)
        print("- full_data saved")

        # remap customers dfs
        customer = self.dr.get_customer()
        # find the customers without a mapping
        unique_cust = customer[DEFAULT_USER_COL].unique()
        missing_customers = [c for c in unique_cust if c not in raw_new_user_ids_dict]
        last_user_key = np.array(list(new_raw_user_ids_dict.keys())).max() + 1
        missing_user_new_raw = dict(
            zip(
                np.arange(last_user_key, last_user_key + len(missing_customers)),
                missing_customers,
            )
        )
        missing_user_raw_new = {v: k for k, v in missing_user_new_raw.items()}
        # update the dictionaries
        new_raw_user_ids_dict.update(missing_user_new_raw)
        raw_new_user_ids_dict.update(missing_user_raw_new)

        customer[DEFAULT_USER_COL] = customer[DEFAULT_USER_COL].map(
            raw_new_user_ids_dict
        )
        customer.reset_index(drop=True).to_feather(self._CUSTOMER_PATH)
        print("- customers_df saved")

        # remap articles dfs
        article = self.dr.get_articles()
        # find the articles without a mapping
        unique_art = article[DEFAULT_ITEM_COL].unique()
        missing_articles = [c for c in unique_art if c not in raw_new_item_ids_dict]
        last_item_key = np.array(list(new_raw_item_ids_dict.keys())).max() + 1
        missing_item_new_raw = dict(
            zip(
                np.arange(last_item_key, last_item_key + len(missing_articles)),
                missing_articles,
            )
        )
        missing_item_raw_new = {v: k for k, v in missing_item_new_raw.items()}
        # update the dictionaries
        new_raw_item_ids_dict.update(missing_item_new_raw)
        raw_new_item_ids_dict.update(missing_item_raw_new)

        article[DEFAULT_ITEM_COL] = article[DEFAULT_ITEM_COL].map(raw_new_item_ids_dict)
        article.reset_index(drop=True).to_feather(self._ARTICLE_PATH)
        print("- articles_df saved")

        # add the users missing from sample submission
        user_sample_sub = self.dr.get_sample_submission()[DEFAULT_USER_COL].unique()
        last_user_key = np.array(list(new_raw_user_ids_dict.keys())).max() + 1
        missing_customers = [
            c for c in user_sample_sub if c not in raw_new_user_ids_dict
        ]
        missing_user_new_raw = dict(
            zip(
                np.arange(last_user_key, last_user_key + len(missing_customers)),
                missing_customers,
            )
        )
        missing_user_raw_new = {v: k for k, v in missing_user_new_raw.items()}
        # update the dictionaries
        new_raw_user_ids_dict.update(missing_user_new_raw)
        raw_new_user_ids_dict.update(missing_user_raw_new)

        # save mapping dictionaries
        dict_dp = self._MAPPING_DICT_PATH

        assert (
            len(new_raw_user_ids_dict.keys()) == 1_371_980
        ), "Wrong number of users! check dictionaries"

        # users
        with open(dict_dp / "raw_new_user_ids_dict.pkl", "wb+") as f:
            pickle.dump(raw_new_user_ids_dict, f)
        with open(dict_dp / "new_raw_user_ids_dict.pkl", "wb+") as f:
            pickle.dump(new_raw_user_ids_dict, f)
        print("- user mapping dicts saved")

        # items
        with open(dict_dp / "raw_new_item_ids_dict.pkl", "wb+") as f:
            pickle.dump(raw_new_item_ids_dict, f)
        with open(dict_dp / "new_raw_item_ids_dict.pkl", "wb+") as f:
            pickle.dump(new_raw_item_ids_dict, f)
        print("- items mapping dicts saved")

    def create_holdin_holdout(self) -> None:
        fd = self.get_full_data()
        hold_in = fd[(fd["t_dat"] <= "2020-09-14")]
        hold_out = fd[(fd["t_dat"] > "2020-09-14")]

        # save holdin holdout
        hold_in.reset_index(drop=True).to_feather(self._HOLDIN_PATH)
        hold_out.reset_index(drop=True).to_feather(self._HOLDOUT_PATH)
        print("- done")


if __name__ == "__main__":
    dataset = LWLWDataset()
    dataset.remap_user_item_ids()
    dataset.create_holdin_holdout()
