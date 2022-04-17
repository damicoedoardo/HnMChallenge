import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.utils.logger import set_color

# load env variables
load_dotenv()
logger = logging.getLogger(__name__)


class DatasetInterface(ABC):
    _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))
    _SUBMISSION_FOLDER = Path(__file__).parent.parent / "submission"

    DATASET_NAME = None
    _DATASET_DESCRIPTION = None

    _DATASET_PATH = None
    _MAPPING_DICT_PATH = None

    _ARTICLE_PATH = None
    _CUSTOMER_PATH = None
    _HOLDIN_PATH = None
    _HOLDOUT_PATH = None
    _FULL_DATA_PATH = None

    # this should be hardcoded
    _ARTICLES_NUM = None
    _CUSTOMERS_NUM = None

    def __init__(self) -> None:
        assert self.DATASET_NAME is not None, "Dataset name has not been set"

        # create and store the dataset description
        self._DATASET_DESCRIPTION = self.create_dataset_description()
        # create the dataset path
        self._DATASET_PATH = self._DATA_PATH / "datasets" / self.DATASET_NAME
        # create the dataset folder
        self._DATASET_PATH.mkdir(parents=True, exist_ok=True)
        self._SUBMISSION_FOLDER.mkdir(parents=True, exist_ok=True)

        # fill paths
        self._MAPPING_DICT_PATH = Path(self._DATASET_PATH / Path("mapping_dict"))
        self._MAPPING_DICT_PATH.mkdir(parents=True, exist_ok=True)

        self._ARTICLE_PATH = Path(self._DATASET_PATH / Path("articles.feather"))
        self._CUSTOMER_PATH = Path(self._DATASET_PATH / Path("customers.feather"))
        self._HOLDIN_PATH = Path(self._DATASET_PATH / Path("holdin.feather"))
        self._HOLDOUT_PATH = Path(self._DATASET_PATH / Path("holdout.feather"))
        self._FULL_DATA_PATH = Path(self._DATASET_PATH / Path("full_data.feather"))
        self._CANDIDATE_ITEMS_PATH = Path(
            self._DATASET_PATH / Path("candidate_items.feather")
        )

        self.dr = DataReader()

    def __str__(self):
        """Rerturn the description of the dataset"""
        return self._DATASET_DESCRIPTION

    def get_dataset_path(self) -> Path:
        return self._DATASET_PATH

    @abstractmethod
    def remap_user_item_ids() -> None:
        """
        Remap user item ids on transaction customer and articles df and creates the mapping dictionary
        - store full_data
        - create and save mapping dict for user and item ids
        - create customer and articles df
        - update the user and item dictionaries with the missing users and items
        """
        pass

    @abstractmethod
    def create_holdin_holdout() -> None:
        """Create holdin and holdout and save them"""
        pass

    @abstractmethod
    def create_dataset_description(self) -> str:
        """Create the dataset description"""
        pass

    def get_holdin(self) -> pd.DataFrame:
        """Return the holdin for the dataset"""
        df = pd.read_feather(self._HOLDIN_PATH)
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        return df

    def get_holdout(self) -> pd.DataFrame:
        """Return the holdout for the dataset"""
        df = pd.read_feather(self._HOLDOUT_PATH)
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        return df

    def get_full_data(self) -> pd.DataFrame:
        """Return the full dataset"""
        df = pd.read_feather(self._FULL_DATA_PATH)
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        return df

    def get_customers_df(self) -> pd.DataFrame:
        """Return the customer df with user feature"""
        df = pd.read_feather(self._CUSTOMER_PATH)
        return df

    def get_articles_df(self) -> pd.DataFrame:
        """Return the article df with item feature"""
        df = pd.read_feather(self._ARTICLE_PATH)
        return df

    def get_raw_new_mapping_dict(self) -> Tuple[dict, dict]:
        """Return the RAW -> NEW mapping dict"""
        uids_p = self._MAPPING_DICT_PATH / "raw_new_user_ids_dict.pkl"
        iids_p = self._MAPPING_DICT_PATH / "raw_new_item_ids_dict.pkl"
        with open(uids_p, "rb") as f:
            uids_dict = pickle.load(f)
        with open(iids_p, "rb") as f:
            iids_dict = pickle.load(f)
        return uids_dict, iids_dict

    def get_new_raw_mapping_dict(self) -> Tuple[dict, dict]:
        """Return the NEW -> RAW mapping dict"""
        uids_p = self._MAPPING_DICT_PATH / "new_raw_user_ids_dict.pkl"
        iids_p = self._MAPPING_DICT_PATH / "new_raw_item_ids_dict.pkl"
        with open(uids_p, "rb") as f:
            uids_dict = pickle.load(f)
        with open(iids_p, "rb") as f:
            iids_dict = pickle.load(f)
        return uids_dict, iids_dict

    def get_candidate_items(self) -> np.ndarray:
        raise NotImplementedError("Not implemented for this dataset!")

    def create_submission(self, recs_df: pd.DataFrame, sub_name: str) -> None:

        assert DEFAULT_USER_COL in recs_df.columns, f"Missing col: {DEFAULT_USER_COL}"

        user_map_dict, item_map_dict = self.get_new_raw_mapping_dict()

        grp_recs_df = recs_df.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
        grp_recs_df = grp_recs_df.to_frame().reset_index()
        # map back to original ids
        grp_recs_df[DEFAULT_USER_COL] = grp_recs_df[DEFAULT_USER_COL].apply(
            lambda x: user_map_dict.get(x)
        )
        grp_recs_df[DEFAULT_ITEM_COL] = grp_recs_df[DEFAULT_ITEM_COL].apply(
            lambda x: " ".join(list(map(item_map_dict.get, x)))
        )
        grp_recs_df = grp_recs_df.rename(
            columns={DEFAULT_ITEM_COL: DEFAULT_PREDICTION_COL}
        )

        final_recs = grp_recs_df

        # check both customer id and item id are in str format
        assert isinstance(
            final_recs.head()[DEFAULT_USER_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        assert isinstance(
            final_recs.head()[DEFAULT_PREDICTION_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        final_recs.to_csv(str(self._SUBMISSION_FOLDER / sub_name) + ".csv", index=False)
        logger.info(
            set_color(
                f"Submission with Filtered Data: {sub_name} created succesfully!",
                "yellow",
            )
        )
