import logging
from ensurepip import version
from functools import reduce
from pathlib import Path

import pandas as pd

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.features.item_features import *
from hnmchallenge.features.user_features import *
from hnmchallenge.features.user_item_features import *
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color

logger = logging.getLogger(__name__)


class FeatureManager:
    _USER_FEATURES = [
        Active,
        Age,
        ClubMemberStatus,
        FashionNewsFrequency,
        Fn,
        AvgPrice,
        UserTendency,
    ]
    _ITEM_FEATURES = [
        ColourGroupCode,
        DepartmentNO,
        GarmentGroupName,
        GraphicalAppearanceNO,
        IndexCode,
        IndexGroupName,
        ItemCount,
        ItemCountLastMonth,
        NumberBought,
        PerceivedColourMasterID,
        PerceivedColourValueID,
        ProductGroupName,
        ProductTypeNO,
        SectionNO,
        Price,
        SalesFactor,
    ]
    _USER_ITEM_FEATURES = [
        # SalesChannel,
        TimeScore,
        TimeWeight,
        TimesItemBought,
        # UserItemSalesFactor,
    ]

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

    def create_features_df(self, name: str, dataset_version: int) -> None:
        # load base df
        base_df = RecsInterface.load_recommendations(name, self.kind)

        # rename the recs column accordingly
        # if it is an `ensemble model` we have "recs" column
        # if it is a `single model` we have "recom_name_recs" column
        cols = base_df.columns
        col_to_rename = [col for col in cols if "recs" in col]
        assert len(col_to_rename) == 1, "More recs column, one is needed!"
        col_to_rename = col_to_rename[0]
        base_df = base_df.rename({col_to_rename: DEFAULT_ITEM_COL}, axis=1)

        if len(self._ITEM_FEATURES) > 0:
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
            # join item features on base df
            print(
                set_color(
                    "Merging item features with base df, can take time...", "yellow"
                )
            )
            base_df = pd.merge(
                base_df, item_features_df, on=DEFAULT_ITEM_COL, how="left"
            )

        if len(self._USER_FEATURES) > 0:
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
            # join user features on base df
            print(
                set_color(
                    "Merging user features with base df, can take time...", "yellow"
                )
            )
            base_df = pd.merge(
                base_df, user_features_df, on=DEFAULT_USER_COL, how="left"
            )

        if len(self._USER_ITEM_FEATURES) > 0:
            # load user-item features (context)
            user_item_features_list = []
            print("Loading user_item features...")
            for user_item_f_class in self._USER_ITEM_FEATURES:
                user_item_f = user_item_f_class(self.dataset, self.kind)
                f = user_item_f.load_feature()
                user_item_features_list.append(f)
            print("join user item features...")
            user_item_features_df = reduce(
                lambda x, y: pd.merge(
                    x, y, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="outer"
                ),
                user_item_features_list,
            )
            # join item and user features on base df
            print(
                set_color(
                    "Merging context features with base df, can take time...", "yellow"
                )
            )
            base_df = pd.merge(
                base_df,
                user_item_features_df,
                on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
                how="left",
            )

        # save the the feature dataset
        dir_path = Path(f"hnmchallenge/models_prediction/dataset_logs/{name}")
        dir_path.mkdir(parents=True, exist_ok=True)

        log_filename = dir_path / f"{name}_{dataset_version}.log"
        log_format = "%(levelname)s %(asctime)s - %(message)s"
        # pass the correct handlers depending on if you want to write a log file or not
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode="w+"),
        ]
        logging.basicConfig(level=logging.INFO, handlers=handlers, format=log_format)
        logger = logging.getLogger(__name__)

        # calculate the correct number of features of the dataset
        number_of_features = (
            len(base_df.columns) - 3
            if self.kind == "train"
            else len(base_df.columns) - 2
        )

        if self.kind == "train":
            # log features used in the dataset only when creating the train version of the dataset
            logger.info(f"Creating Dataset: {name}_{dataset_version}")
            logger.info(f"Number of features: {number_of_features}")

            logger.info("\n\nUSER FEATURES:\n")
            for u_f in self._USER_FEATURES:
                logger.info(f"{u_f.FEATURE_NAME}")

            logger.info("\n\nITEM FEATURES:\n")
            for i_f in self._ITEM_FEATURES:
                logger.info(f"{i_f.FEATURE_NAME}")

            logger.info("\n\nCONTEXT FEATURES:\n")
            for u_i_f in self._USER_ITEM_FEATURES:
                logger.info(f"{u_i_f.FEATURE_NAME}")

        # save features df
        dir_path = dr.get_preprocessed_data_path() / Path(f"dataset_dfs/{self.kind}")
        dir_path.mkdir(parents=True, exist_ok=True)
        save_name = f"{name}_{dataset_version}.feather"
        base_df.reset_index(drop=True).to_feather(dir_path / save_name)
        print(f"Dataset saved succesfully in : {dir_path / save_name}")


if __name__ == "__main__":
    KIND = "train"
    DATASET_NAME = "cutf_100_TimePop_alpha_0.9"
    VERSION = 0

    dr = DataReader()
    dataset = StratifiedDataset()
    fm = FeatureManager(dataset, KIND)
    fm.create_features_df(DATASET_NAME, VERSION)
