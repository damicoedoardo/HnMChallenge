import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color


class RecsInterface(ABC):
    """Interface to create ensemble recs from different models"""

    # Name of the model to be considered in the ensemble
    RECS_NAME = None
    _SAVE_PATH = "recommendations_dfs"

    def __init__(
        self, kind: str, dataset: StratifiedDataset, cutoff: int = 100
    ) -> None:
        assert kind in ["train", "full"], "kind should be train or full"
        self.kind = kind
        self.dataset = dataset
        self.dr = DataReader()
        self.cutoff = cutoff
        # creating recs directory
        self.save_path = (
            self.dr.get_preprocessed_data_path() / self._SAVE_PATH / self.kind
        )
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _check_recommendations_integrity(self, recs: pd.DataFrame) -> None:
        """Check the integrity of the reocmmendations passed"""
        columns = recs.columns
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
    def get_recommendations(self) -> pd.DataFrame:
        """Generate recommendation with a given cutoff"""
        pass

    def save_recommendations(self) -> None:
        """Retrieve recommendations and save them
        is self.kind == "train" the `relevance` column is added
        """
        print(
            set_color(
                f"Kind: {self.kind}, Cutoff: {self.cutoff}, retrieving and saving recs...",
                color="cyan",
            )
        )
        recs = self.get_recommendations()
        self._check_recommendations_integrity(recs)

        if self.kind == "train":
            print("Creating Relevance column...")
            # loading holdout groundtruth
            holdout_groundtruth = self.dataset.get_holdout_groundtruth()

            # merge recs and item groundtruth
            merged = pd.merge(
                recs,
                holdout_groundtruth,
                left_on=[DEFAULT_USER_COL, f"{self.RECS_NAME}_recs"],
                right_on=[DEFAULT_USER_COL, "article_id"],
                how="left",
            )

            # we have to remove the user for which we do not do at least one hit,
            # since we would not have the relavance for the items
            merged.loc[merged["article_id"].notnull(), "article_id"] = 1
            merged["hit_sum"] = merged.groupby(DEFAULT_USER_COL)[
                "article_id"
            ].transform("sum")

            merged_filtered = merged[merged["hit_sum"] > 0]

            # we can drop the hit sum column
            merged_filtered = merged_filtered.drop("hit_sum", axis=1)

            # fill with 0 the nan values, the nan are the one for which we do not do an hit
            merged_filtered["article_id"] = merged_filtered["article_id"].fillna(0)

            # rename the columns
            merged_filtered = merged_filtered.rename(
                {"article_id": "relevance"}, axis=1
            ).reset_index(drop=True)
            recs = merged_filtered
            print("Done!")

        # save the retrieved recommendations
        save_name = f"{self.kind}_cutf_{self.cutoff}_{self.RECS_NAME}.feather"
        recs.reset_index(drop=True).to_feather(self.save_path / save_name)

    def load_recommendations(self) -> pd.DataFrame:
        """Load recommendations"""
        print(set_color(f"Cutoff: {self.cutoff}, loading recs...", color="cyan"))
        load_name = f"{self.kind}_cutf_{self.cutoff}_{self.RECS_NAME}"
        load_path = self.save_path / load_name
        recs = pd.read_feather(load_path)
        return recs

    def eval_recommendations(self, write_log: bool = True) -> None:
        """Evaluate recommendations on holdout set"""
        assert (
            self.kind == "train"
        ), "To evaluate recommendation `kind` should be `train`"

        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        log_filename = f"hnmchallenge/models_prediction/evaluation_logs/{dt_string}.log"

        log_format = "%(levelname)s %(asctime)s - %(message)s"
        # pass the correct handlers depending on if you want to write a log file or not
        if write_log:
            handlers = [
                logging.StreamHandler(),
                logging.FileHandler(log_filename, mode="w+"),
            ]

        else:
            handlers = [
                logging.StreamHandler(),
            ]

        logging.basicConfig(level=logging.INFO, handlers=handlers, format=log_format)
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluating: {self.RECS_NAME}, cutoff:{self.cutoff} \n")

        # retrieve recs
        recs = self.get_recommendations()
        self._check_recommendations_integrity(recs)

        # load groundtruth and holdout data
        holdout_groundtruth = self.dataset.get_holdout_groundtruth()
        holdout = self.dataset.get_last_month_holdout()

        # merge recs and item groundtruth
        merged = pd.merge(
            recs,
            holdout_groundtruth,
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
        logger.info(f"MAP@{self.cutoff}: {map_at_k(ground_truth, pred)}")
        logger.info(f"RECALL@{self.cutoff}: {recall_at_k(ground_truth, pred)}")
        logger.info("\nMetrics on ONE-HIT users")
        logger.info(f"MAP@{self.cutoff}: {map_at_k(ground_truth, pred_filtered)}")
        logger.info(
            f"RECALL@{self.cutoff}: {recall_at_k(ground_truth, pred_filtered)}",
        )
