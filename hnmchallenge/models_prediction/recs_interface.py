import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.utils.logger import set_color


class RecsInterface(ABC):
    """Interface to create ensemble recs from different models"""

    # Name of the model to be considered in the ensemble
    RECS_NAME = None
    _SAVE_PATH = "recommendations_dfs"

    def __init__(self, kind: str, dataset, cutoff: Union[int, list] = 100) -> None:
        assert kind in ["train", "full"], "kind should be train or full"
        self.kind = kind
        self.dataset = dataset
        self.dr = DataReader()
        self.cutoff = cutoff
        # creating recs directory
        self.save_path = self.dataset._DATASET_PATH / self._SAVE_PATH / self.kind
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

    def _add_pop_missing_users(self, recs_df, num_items=100) -> pd.DataFrame:
        assert (
            self.kind == "full"
        ), "this function has be called only for the full recommendations!"
        u_md, _ = self.dataset.get_new_raw_mapping_dict()
        all_users = set(np.array(list(u_md.keys())))
        recs_users = set(recs_df[DEFAULT_USER_COL].unique())
        missing_users = all_users.difference(recs_users)
        missing_users_df = pd.DataFrame(list(missing_users), columns=[DEFAULT_USER_COL])

        # compute popularity
        fd = self.dataset.get_full_data()
        count_mb = fd.groupby(DEFAULT_ITEM_COL).count()
        feature = count_mb.reset_index()[[DEFAULT_ITEM_COL, "t_dat"]].rename(
            columns={"t_dat": "popularity"}
        )
        feature["popularity_score"] = (
            feature["popularity"] - feature["popularity"].min()
        ) / (feature["popularity"].max() - feature["popularity"].min())
        feature["rank"] = (
            feature["popularity_score"].rank(ascending=False, method="min").astype(int)
        )
        missing_users_df["temp"] = 1
        feature = feature[feature["rank"] <= num_items]
        feature = feature.sort_values("rank")
        feature["temp"] = 1
        final_df = pd.merge(missing_users_df, feature, on="temp")
        missing_recs = final_df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]

        col_name = [c for c in recs_df.columns if "recs" in c][0]
        missing_recs = missing_recs.rename({DEFAULT_ITEM_COL: col_name}, axis=1)

        # concat with the other recommendations
        final_recs = pd.concat([recs_df, missing_recs], axis=0)
        return final_recs

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

            # retrieve the holdout
            holdout = self.dataset.get_holdout()
            # retrieve items per user in holdout
            item_per_user = holdout.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(
                list
            )
            item_per_user_df = item_per_user.to_frame()
            # items groundtruth
            holdout_groundtruth = (
                item_per_user_df.reset_index()
                .explode(DEFAULT_ITEM_COL)
                .drop_duplicates()
            )

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

        if self.kind == "full":
            print("Adding pop predictions on missing users")
            recs = self._add_pop_missing_users(recs)

        # save the retrieved recommendations
        save_name = f"cutf_{self.cutoff}_{self.RECS_NAME}.feather"
        recs.reset_index(drop=True).to_feather(self.save_path / save_name)

    @staticmethod
    def load_recommendations(dataset, name: str, kind: str) -> pd.DataFrame:
        """Load recommendations"""
        assert kind in ["train", "full"], "`kind` should be in train or full"
        print(set_color(f"loading recs model:\n {name}", color="cyan"))

        save_path = dataset._DATASET_PATH / "recommendations_dfs" / kind
        load_name = f"{name}.feather"
        load_path = save_path / load_name
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
        log_name = self.RECS_NAME + "__" + dt_string

        dir_path = Path("hnmchallenge/models_prediction/evaluation_logs")
        log_filename = dir_path / f"{log_name}.log"
        print(log_filename)

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
        logger.info(
            f"Dataset: {self.dataset.DATASET_NAME},\n description:{print(self.dataset)} \n"
        )
        logger.info(f"Evaluating: {self.RECS_NAME}, cutoff:{self.cutoff} \n")

        # retrieve recs
        recs = self.get_recommendations()
        self._check_recommendations_integrity(recs)

        # retrieve the holdout
        holdout = self.dataset.get_holdout()
        # retrieve items per user in holdout
        item_per_user = holdout.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
        item_per_user_df = item_per_user.to_frame()
        # items groundtruth
        holdout_groundtruth = (
            item_per_user_df.reset_index().explode(DEFAULT_ITEM_COL).drop_duplicates()
        )

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
        ground_truth = holdout_groundtruth[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].copy()
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
