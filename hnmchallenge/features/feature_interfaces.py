import logging
import os
from abc import ABC, abstractmethod

import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.decorator import timing
from hnmchallenge.utils.logger import set_color


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
    def _check_integrity(self, featue: pd.DataFrame) -> None:
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
        self._check_integrity(feature)
        print(f"Saving Feature {self.kind}...")
        feature_name = self.FEATURE_NAME + ".feather"
        feature.reset_index(drop=True).to_feather(self.save_path / feature_name)
        print(f"Feature saved in {self.save_path / self.FEATURE_NAME}")

    def load_feature(self) -> pd.DataFrame:
        name_f = self.FEATURE_NAME + ".feather"
        feature = pd.read_feather(self.save_path / name_f)
        return feature


class UserItemFeature(Feature):
    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _check_integrity(self, feature: pd.DataFrame) -> None:
        assert (
            DEFAULT_ITEM_COL in feature.columns
        ), f"{DEFAULT_ITEM_COL} not in feature columns"
        assert (
            DEFAULT_USER_COL in feature.columns
        ), f"{DEFAULT_USER_COL} not in feature columns"
        assert "t_dat" in feature.columns, f"t_dat not in feature columns"

        keys_df = self._get_keys_df()

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
    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _check_integrity(self, feature: pd.DataFrame) -> None:
        assert (
            DEFAULT_ITEM_COL in feature.columns
        ), f"{DEFAULT_ITEM_COL} not in feature columns"
        keys_df = self._get_keys_df()

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
        item_df = (
            self.dr.get_filtered_articles()[DEFAULT_ITEM_COL]
            .to_frame()
            .drop_duplicates()
        )
        return item_df


class UserFeature(Feature):
    def __init__(self, dataset: StratifiedDataset, kind: str) -> None:
        super().__init__(dataset, kind)

    def _check_integrity(self, feature: pd.DataFrame) -> None:
        assert (
            DEFAULT_USER_COL in feature.columns
        ), f"{DEFAULT_USER_COL} not in feature columns"
        keys_df = self._get_keys_df()

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
        user_df = (
            self.dr.get_filtered_customers()[DEFAULT_USER_COL]
            .to_frame()
            .drop_duplicates()
        )
        return user_df
