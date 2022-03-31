import pickle

import numpy as np
from hnmchallenge.dataset_interface import DatasetInterface


class LMLWDataset(DatasetInterface):

    DATASET_NAME = "LMLW_dataset"

    def __init__(self) -> None:
        super().__init__()

    def create_dataset_description(self) -> str:
        description = """ 
        Items: t_dat > 01/09/2020 \n
        holdout: last_week
        """
        return description

    def remap_user_item_ids(self) -> None:
        tr = self.dr.get_transactions()

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
        df_name = "full_data.feather"
        tr.to_feather(self._DATASET_PATH / df_name)

        # save mapping dictionaries
        dict_dp = self._MAPPING_DICT_PATH

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

        # user df

    def create_holdin_holdout(self) -> None:
        pass
