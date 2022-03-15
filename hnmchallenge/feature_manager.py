import logging
import os
from abc import ABC, abstractmethod
from functools import reduce

import pandas as pd
from pyexpat import features

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.features import user_features
from hnmchallenge.features.item_features import *
from hnmchallenge.features.user_features import *
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.decorator import timing
from hnmchallenge.utils.logger import set_color

logger = logging.getLogger(__name__)


class FeatureManager:
    _USER_FEATURES = [
        Active,
        Age,
        ClubMemberStatus,
        FashionNewsFrequency,
        Fn,
    ]
    _ITEM_FEATURES = [
        ColourGroupCode,
        DepartmentNO,
        GarmentGroupName,
        GraphicalAppearanceNO,
        IndexCode,
        IndexGroupName,
        PerceivedColourMasterID,
        PerceivedColourValueID,
        ProductGroupName,
        ProductTypeNO,
        SectionNO,
    ]
    _USER_ITEM_FEATURES = []

    def __init__(
        self,
        dataset: StratifiedDataset,
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
        self.kind = kind
        self.dr = DataReader()
        self.dataset = dataset

    def create_features_df(self, name: str) -> pd.DataFrame:
        # if kind is "train" means we need the relevance df,
        # if kind is "full" we need recommendations df
        # load base df
        kind_name = self.kind + "_" + name
        if self.kind == "train":
            base_df_path = (
                self.dr.get_preprocessed_data_path() / "relevance_dfs" / kind_name
            )
            print("Creating features df for training...")
        else:
            base_df_path = self.dr.get_preprocessed_data_path() / kind_name
            print("Creating features df for final predictions...")
        base_df = pd.read_feather(base_df_path)

        # load item features
        item_features_list = []
        print("Loading item features...")
        for item_f_class in self._ITEM_FEATURES:
            item_f = item_f_class(self.dataset, self.kind)
            f = item_f.load_feature()
            item_features_list.append(f)
        print("join item features...")
        # joining item features
        item_features_df = reduce(
            lambda x, y: pd.merge(x, y, on=DEFAULT_ITEM_COL, how="outer"),
            item_features_list,
        )

        # load user features
        user_features_list = []
        print("Loading user features...")
        for user_f_class in self._USER_FEATURES:
            user_f = user_f_class(self.dataset, self.kind)
            f = user_f.load_feature()
            user_features_list.append(f)
        print("join user features...")
        user_features_df = reduce(
            lambda x, y: pd.merge(x, y, on=DEFAULT_USER_COL, how="outer"),
            user_features_list,
        )

        # join item and user features on base df
        base_df = pd.merge(base_df, item_features_df, on=DEFAULT_ITEM_COL, how="left")
        base_df = pd.merge(base_df, user_features_df, on=DEFAULT_USER_COL, how="left")

        print(f"Final number of features loaded: {len(base_df.columns) - 3}")
        return base_df
