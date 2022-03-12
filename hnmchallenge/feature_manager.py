import logging
import os
from abc import ABC, abstractmethod

import pandas as pd
from pyexpat import features

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.decorator import timing
from hnmchallenge.utils.logger import set_color

logger = logging.getLogger(__name__)


class FeatureManager:
    _USER_FEATURES = []
    _ITEM_FEATURES = []
    _USER_ITEM_FEATURES = ["item_price"]

    def __init__(
        self,
        kind: str,
        user_features: list[str] = None,
        item_features: list[str] = None,
        user_item_features: list[str] = None,
    ) -> None:
        # check correctness features passed
        if user_features is not None:
            for userf in user_features:
                assert (
                    userf in self._USER_FEATURES
                ), f"feature {userf} not in _USER_FEATURES, add it on feature_manager.py class!"

        if item_features is not None:
            for itemf in item_features:
                assert (
                    itemf in self._ITEM_FEATURES
                ), f"feature {itemf} not in _ITEM_FEATURES, add it on feature_manager.py class!"

        if user_item_features is not None:
            for useritemf in user_item_features:
                assert (
                    useritemf in self._USER_ITEM_FEATURES
                ), f"feature {useritemf} not in _USER_ITEM_FEATURES, add it on feature_manager.py class!"

        # if not feature passed meaning we use all of them
        if user_features is None:
            logger.info(set_color(f"Using ALL available User features", "cyan"))
            user_features = self._USER_FEATURES
        if item_features is None:
            logger.info(set_color(f"Using ALL available Item features", "cyan"))
            item_features = self._ITEM_FEATURES
        if user_item_features is None:
            logger.info(set_color(f"Using ALL available Uaer Item features", "cyan"))
            user_item_features = self._USER_ITEM_FEATURES

        self.user_features = user_features
        self.item_features = item_features
        self.user_item_features = user_item_features

    def create_features_df(self):
        # 1) load the relevance df, pass the name of which relevance df has to be used
        # 2) join the user-item features
        # 3) join item features
        # 4) joim user features
        pass

    def create_libsvm_dataset():
        # TODO: account for train val test, name of the dataset, kind, save as binary file, create a new folder for libsvm files
        pass


class Feature(ABC):
    _SAVE_PATH = "features"
    _KINDS = ["train", "full"]
    FEATURE_NAME = None

    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        self.dataset = dataset
        assert kind in self._KINDS, f"kind should be in {self._KINDS}"
        self.kind = kind
        self.dr = DataReader()

        # creating features directory
        self.save_path = (
            self.dr.get_preprocessed_data_path() / self._SAVE_PATH / self.kind
        )
        self.save_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _check_integrity(self) -> None:
        """Check integrity of a feature"""
        # TODO check no nan, check all rows mantained
        pass

    @abstractmethod
    def _create_feature(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_keys_df() -> pd.DataFrame:
        """Return the skeleton dataframe to build a new feature

        Returns:
            pd.DataFrane: skeleton dataframe with the keys columns
        """
        pass

    @timing
    def save_feature(self) -> None:
        feature = self._create_feature()
        self._check_integrity()
        print(f"Saving Feature {self.kind}...")
        feature_name = self.FEATURE_NAME + ".feather"
        feature.reset_index(drop=True).to_feather(self.save_path / feature_name)
        print(f"Feature saved in {self.save_path / self.FEATURE_NAME}")


class UserItemFeature(Feature):
    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _check_integrity(self) -> None:
        keys_df = self._get_keys_df()
        feature = self._create_feature()

        # check no missing rows
        assert len(keys_df) == len(
            feature
        ), f"Missing rows, given: {len(feature)}\n wanted: {len(keys_df)}\n"
        # check no missing values
        assert (
            not feature.isnull().values.any()
        ), "NaN values present, please fill them in _create_feature()"
        assert self.FEATURE_NAME is not None, "feature name has not been set!"

    def _get_keys_df(self) -> pd.DataFrame:
        data_df = None
        if self.kind == "train":
            data_df = self.dataset.get_holdin()
        else:
            data_df = self.dr.get_filtered_full_data()

        data_df = data_df[
            [DEFAULT_USER_COL, DEFAULT_ITEM_COL, "t_dat"]
        ].drop_duplicates()
        return data_df


class ItemFeature(Feature):
    def __init__(self) -> None:
        pass


class UserFeature(Feature):
    def __init__(self) -> None:
        pass
