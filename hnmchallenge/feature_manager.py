import logging
import time
from functools import reduce
from pathlib import Path

import pandas as pd

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.datasets.all_items_last_mont__last_day_last_week import AILMLDWDataset
from hnmchallenge.datasets.all_items_last_month_last_day import AILMLDDataset
from hnmchallenge.datasets.all_items_last_month_last_day_last_2nd_week import (
    AILMLD2WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_day_last_3rd_week import (
    AILMLD3WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_day_last_4th_week import (
    AILMLD4WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_day_last_5th_week import (
    AILMLD5WDataset,
)
from hnmchallenge.datasets.all_items_last_month_last_week import AILMLWDataset
from hnmchallenge.datasets.last2month_last_day import L2MLDDataset
from hnmchallenge.datasets.last_month_last_2nd_week_dataset import LML2WDataset
from hnmchallenge.datasets.last_month_last_3rd_week_dataset import LML3WDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_month_last_week_user import LMLUWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.features.item_features import *
from hnmchallenge.features.light_gbm_features import *
from hnmchallenge.features.user_features import *
from hnmchallenge.features.user_item_features import *
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.utils.logger import set_color

logger = logging.getLogger(__name__)


class FeatureManager:
    _GBM_FEATURES = [
        GraphicalAppearanceNOGBM,
        IndexCodeGBM,
        IndexGroupNameGBM,
        ProductGroupNameGBM,
    ]
    _USER_FEATURES = [
        LastBuyDate,
        TotalItemsBought,
        Active,
        Age,
        ClubMemberStatus,
        FashionNewsFrequency,
        Fn,
        AvgPrice,
        UserTendency,
        UserTendencyLM,
        UserAvgBuyDay,
        SaleChannelScore,
        UserAvgBuySession,
    ]
    _ITEM_FEATURES = [
        PopularityCumulative,
        ColourGroupCode,
        ItemSaleChannelScore,
        DepartmentNO,
        ##GarmentGroupName,
        # GraphicalAppearanceNO,
        GarmentGroupNO,
        # IndexCode,
        # IndexGroupName,
        IndexGroupNO,
        ItemCount,
        ItemCountLastMonth,
        NumberBought,
        PerceivedColourMasterID,
        PerceivedColourValueID,
        # ProductGroupName,
        ProductTypeNO,
        SectionNO,
        Price,
        SalesFactor,
        ItemSaleChannelScore,
        PopSales1,
        PopSales2,
        ItemAgePop,
        ItemPriceProduct,
    ]
    _USER_ITEM_FEATURES = [
        TimeScore,
        TimeWeight,
        TimesItemBought,
        # ItemKNNScore,
    ]

    def __init__(
        self,
        dataset,
        kind: str,
        user_features: list[str] = None,
        item_features: list[str] = None,
        user_item_features: list[str] = None,
        gbm_features: list[str] = None,
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

        if gbm_features is not None:
            for gbmf in gbm_features:
                assert (
                    gbmf in self._GBM_FEATURES
                ), f"feature {gbmf} not in _GBM_FEATURES, add it on feature_manager.py class!"

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
        if gbm_features is None:
            logger.info(set_color(f"Using ALL available GBM features", "cyan"))
            gbm_features = self._GBM_FEATURES

        self.user_features = user_features
        self.item_features = item_features
        self.user_item_features = user_item_features
        self.gbm_features = gbm_features
        self.kind = kind
        self.dr = DataReader()
        self.dataset = dataset

    def create_features_df(self, name: str, dataset_version: int) -> None:
        # load base df

        base_df = RecsInterface.load_recommendations(self.dataset, name, self.kind)

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
            # s = time.time()
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
            # print(f"Taken: {time.time()-s}")

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
        if len(self._GBM_FEATURES) > 0:
            # load item features
            gbm_features_list = []
            print("Loading GBM features...")
            for gbm_f_class in self._GBM_FEATURES:
                gbm_f = gbm_f_class(self.dataset, self.kind)
                f = gbm_f.load_feature()
                gbm_features_list.append(f)
            print("join GBM features...")

            # joining item features
            gbm_features_df = reduce(
                lambda x, y: pd.merge(x, y, on=DEFAULT_ITEM_COL, how="outer"),
                gbm_features_list,
            )

            # join item features on base df
            print(
                set_color(
                    "Merging gbm item features with base df, can take time...", "yellow"
                )
            )
            base_df = pd.merge(
                base_df, gbm_features_df, on=DEFAULT_ITEM_COL, how="left"
            )

        ###################
        # augment features
        ###################
        # fill nan values
        # print("Augmenting features...")

        # # try to create features to asses if is the case to predict an item that the user has just bought
        # base_df["times_item_bought"] = base_df["times_item_bought"].fillna(0)
        # base_df["tdiff"] = base_df["tdiff"].fillna(1)
        # # compute augmented features on tendency multiple buy
        # base_df["mb_stats_tdiff"] = (1 - base_df["tdiff"]) * base_df["user_tendency"]
        # base_df["mb_stats_number_bought"] = (
        #     base_df["times_item_bought"] * base_df["user_tendency"]
        # )

        ##############################
        # save the the feature dataset
        ##############################
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

            logger.info("\n\nGBM FEATURES:\n")
            for g_f in self._GBM_FEATURES:
                logger.info(f"{g_f.FEATURE_NAME}")

        # save features df
        dir_path = self.dataset._DATASET_PATH / Path(f"dataset_dfs/{self.kind}")
        dir_path.mkdir(parents=True, exist_ok=True)
        save_name = f"{name}_{dataset_version}.feather"
        base_df.reset_index(drop=True).to_feather(dir_path / save_name)
        print(f"Dataset saved succesfully in : {dir_path / save_name}")


if __name__ == "__main__":
    # KIND = "train"
    # DATASET_NAME = "cutf_200_TimePop_alpha_1.0"
    DATASET_NAME = f"cutf_300_ItemKNN_tw_True_rs_False"
    # DATASET_NAME = "cutf_100_TimePop_alpha_1.0"
    # DATASET_NAME = "dataset_v1000"
    VERSION = 0

    # dataset = LMLWDataset()
    DATASETS = [AILMLD4WDataset(), AILMLD5WDataset()]
    for dataset in DATASETS:
        s = time.time()
        for kind in ["train"]:
            # for kind in ["train", "full"]:
            # for kind in ["full"]:
            dr = DataReader()
            fm = FeatureManager(dataset, kind)
            fm.create_features_df(DATASET_NAME, VERSION)
        print(f"took: {time.time()-s}s")
