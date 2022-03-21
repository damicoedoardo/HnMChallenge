import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color


class RecsEnsembleInterface(ABC):
    """Interface to create ensemble recs from different models"""

    # Name of the model to be considered in the ensemble
    RECS_NAME = None
    _SAVE_PATH = "recommendations_dfs"

    def __init__(self, kind: str, dataset: StratifiedDataset) -> None:
        assert kind in ["train", "full"], "kind should be train or full"
        self.kind = kind
        self.dataset = dataset
        self.dr = DataReader()
        # creating recs directory
        self.save_path = (
            self.dr.get_preprocessed_data_path() / self._SAVE_PATH / self.kind
        )
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _check_recommendations_integrity(self, recs: pd.DataFrame) -> None:
        """Check the integrity of the reocmmendations passed"""
        columns = recs.columns
        assert len(columns) == 4, "Too many columns passed on recs df"
        assert DEFAULT_USER_COL in columns, f"Missing {DEFAULT_USER_COL} on recs df"
        assert any(
            ["recs" in col for col in columns]
        ), f"Missing recs column on recs df"
        assert any(
            ["rank" in col for col in columns]
        ), f"Missing rank column on recs df"
        assert any(
            ["score" in col for col in columns]
        ), f"Missing score column on recs df"

    @abstractmethod
    def get_recommendations(self, cutoff: int = 100) -> pd.DataFrame:
        """Generate recommendation with a given cutoff"""
        pass

    def save_recommendations(self, cutoff: int = 100) -> None:
        """Retrieve recommendations and save them"""
        print(
            set_color(
                f"Kind: {self.kind}, Cutoff: {cutoff}, retrieving and saving recs...",
                color="cyan",
            )
        )

        recs = self.get_recommendations(cutoff=cutoff)
        self._check_recommendations_integrity(recs)

        # save the retrieved recommendations
        save_name = f"{self.kind}_cutf_{cutoff}_{self.RECS_NAME}.feather"
        recs.reset_index(drop=True).to_feather(self.save_path / save_name)

    def load_recommendations(self, cutoff: int = 100) -> pd.DataFrame:
        """Load recommendations"""
        print(set_color(f"Cutoff: {cutoff}, loading recs...", color="cyan"))
        load_name = f"{self.kind}_cutf_{cutoff}_{self.RECS_NAME}"
        load_path = self.save_path / load_name
        recs = pd.read_feather(load_path)
        return recs

    def eval_recommendations(self, cutoff: int = 100) -> None:
        """Evaluate recommendations on holdout set"""
        assert (
            self.kind == "train"
        ), "To evaluate recommendation `kind` should be `train`"

        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        log_filename = f"hnmchallenge/models_prediction/logs/{dt_string}.log"

        log_format = "%(levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_filename, mode="w+"),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluating: {self.RECS_NAME}, cutoff:{cutoff} \n")

        # retrieve recs
        recs = self.get_recommendations(cutoff=cutoff)
        self._check_recommendations_integrity(recs)

        # retrieve holdout data
        holdout = self.dataset.get_last_month_holdout()
        # retrieve items per user in holdout
        item_per_user = holdout.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
        item_per_user_df = item_per_user.to_frame()
        # items groundtruth
        items_groundtruth = (
            item_per_user_df.reset_index().explode(DEFAULT_ITEM_COL).drop_duplicates()
        )
        # merge recs and item groundtruth
        merged = pd.merge(
            recs,
            items_groundtruth,
            left_on=[DEFAULT_USER_COL, f"{self.RECS_NAME}_recs"],
            right_on=[DEFAULT_USER_COL, "article_id"],
            how="left",
        )

        # we have to remove the user for which we do not do at least one hit,
        # since we would not have the relavance for the items
        merged.loc[merged["article_id"].notnull(), "article_id"] = 1
        merged["hit_sum"] = merged.groupby(DEFAULT_USER_COL)["article_id"].transform(
            "sum"
        )

        merged_filtered = merged[merged["hit_sum"] > 0]

        pred = (
            merged[
                [DEFAULT_USER_COL, f"{self.RECS_NAME}_recs", f"{self.RECS_NAME}_rank"]
            ]
            .copy()
            .rename(
                {
                    f"{self.RECS_NAME}_recs": DEFAULT_ITEM_COL,
                    f"{self.RECS_NAME}_rank": "rank",
                },
                axis=1,
            )
        )
        pred_filtered = (
            merged_filtered[
                [DEFAULT_USER_COL, f"{self.RECS_NAME}_recs", f"{self.RECS_NAME}_rank"]
            ]
            .copy()
            .rename(
                {
                    f"{self.RECS_NAME}_recs": DEFAULT_ITEM_COL,
                    f"{self.RECS_NAME}_rank": "rank",
                },
                axis=1,
            )
        )
        ground_truth = holdout[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].copy()
        logger.info(
            f"Remaining Users (at least one hit): {merged_filtered[DEFAULT_USER_COL].nunique()}"
        )
        logger.info("\nMetrics on ALL users")
        logger.info(f"MAP@{cutoff}: {map_at_k(ground_truth, pred)}")
        logger.info(f"RECALL@{cutoff}: {recall_at_k(ground_truth, pred)}")
        logger.info("\nMetrics on ONE-HIT users")
        logger.info(f"MAP@{cutoff}: {map_at_k(ground_truth, pred_filtered)}")
        logger.info(
            f"RECALL@{cutoff}: {recall_at_k(ground_truth, pred_filtered)}",
        )
