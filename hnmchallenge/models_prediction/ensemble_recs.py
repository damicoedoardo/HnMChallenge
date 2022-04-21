import logging
from datetime import datetime
from functools import partial, reduce
from pathlib import Path
from pickle import FALSE

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.datasets.all_items_last_month_last_week import AILMLWDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.models_prediction.bought_items_recs import BoughtItemsRecs
from hnmchallenge.models_prediction.ease_recs import EaseRecs
from hnmchallenge.models_prediction.itemknn_recs import ItemKNNRecs
from hnmchallenge.models_prediction.popularity_recs import PopularityRecs
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.models_prediction.time_pop import TimePop
from hnmchallenge.utils.logger import set_color
from matplotlib.pyplot import axis


class EnsembleRecs(RecsInterface):
    RECS_NAME = None

    def __init__(self, models_list: list, kind: str, dataset) -> None:
        # Check that all the models passed are extending Recs Interface
        assert (
            len(models_list) > 1
        ), "At least 2 models should be passed to create an Ensemble !"

        cutoff = -1
        super().__init__(kind=kind, dataset=dataset, cutoff=-1)

        self.models_list = models_list
        self.RECS_NAME = self._create_ensemble_name()
        self.avg_recs_per_user = None

    def _create_ensemble_name(self):
        """Create a meaningfull name for the ensamble model"""
        ensemble_name = "\n".join(self.models_list)
        print(f"Creating ensemble with:\n{ensemble_name}\n\n")
        return ensemble_name

    def get_recommendations(self) -> pd.DataFrame:
        # load the recommendations for every model
        recs_dfs_list = [
            RecsInterface.load_recommendations(self.dataset, m, kind=self.kind)
            for m in self.models_list
        ]

        def _merge_dfs(kind, df_x, df_y):
            # search for the names on which perform the join
            recs_col_x = [c for c in df_x.columns if "recs" in c][0]
            print(recs_col_x)
            recs_col_y = [c for c in df_y.columns if "recs" in c][0]
            print(recs_col_y)

            LEFT_ON = [DEFAULT_USER_COL, recs_col_x]
            RIGHT_ON = [DEFAULT_USER_COL, recs_col_y]
            if kind == "train":
                # we have to  merge also on the relevance
                LEFT_ON = [DEFAULT_USER_COL, recs_col_x, "relevance"]
                RIGHT_ON = [DEFAULT_USER_COL, recs_col_y, "relevance"]
            print(LEFT_ON)
            print(RIGHT_ON)

            merged = pd.merge(
                df_x,
                df_y,
                left_on=LEFT_ON,
                right_on=RIGHT_ON,
                how="outer",
            )

            merged["recs"] = merged.filter(like="recs").ffill(axis=1).iloc[:, -1]

            # drop the unmerged recs columns
            cols_to_drop = [c for c in merged.columns if "_recs" in c]
            print(f"dropping cols: {cols_to_drop}")
            merged = merged.drop(cols_to_drop, axis=1)

            return merged

        _merge_dfs_kind = partial(_merge_dfs, self.kind)
        merged_recs_df = reduce(_merge_dfs_kind, recs_dfs_list)

        # store average recs per user
        self.avg_recs_per_user = merged_recs_df.groupby(DEFAULT_USER_COL).size().mean()
        print(f"Average recs per user: {self.avg_recs_per_user}")

        # add ensemble_rank column for evaluation
        # the prediction will be sorted using the order of the model given as input 1st in list more important
        ### Can be done but it is very slow, not really needed we do not care about ranking for the recommender systems
        ### we care about Recall so we can drop this part for efficiency
        # models_names = [model.RECS_NAME + "_score" for model in self.models_list]
        # print(set_color("Sorting scores, can take a while...", "yellow"))
        # ensemble_recs = merged_recs_df.sort_values(
        #     [DEFAULT_USER_COL, *models_names], ascending=False
        # )
        # print(set_color("Done!", "yellow"))

        # can be substituted with code above
        ensemble_recs = merged_recs_df.sort_values(DEFAULT_USER_COL)

        recs_per_user = ensemble_recs.groupby(DEFAULT_USER_COL).size().values
        ensemble_rank = np.concatenate(
            list(map(lambda x: np.arange(1, x + 1), recs_per_user))
        )
        # adding final ensemble rank calling it `rank` for evaluation library
        ensemble_recs["rank"] = ensemble_rank

        return ensemble_recs

    def save_recommendations(self, dataset_name: str) -> None:
        # write the log file and so on
        print(
            set_color(
                f"Kind: {self.kind}\nretrieving and saving recs...",
                color="cyan",
            )
        )
        save_name = f"{dataset_name}.feather"
        assert not Path(
            self.save_path / save_name
        ).exists(), f"dataset: {save_name}, exsist, change name."

        recs = self.get_recommendations()
        self._check_recommendations_integrity(recs)

        if self.kind == "full":
            recs = self._add_pop_missing_users(recs)

        # col_to_drop = [col for col in recs.columns if "rank" in col]
        # print(col_to_drop)
        # recs = recs.drop(col_to_drop, axis=1)
        # print(recs.columns)
        recs = recs.drop("rank", axis=1)

        # save the retrieved recommendations
        recs.reset_index(drop=True).to_feather(self.save_path / save_name)

    def eval_recommendations(self, dataset_name: str, write_log: bool = True) -> None:
        """Evaluate recommendations on holdout set"""
        assert (
            self.kind == "train"
        ), "To evaluate recommendation `kind` should be `train`"

        # create log directory
        dir_path = Path(f"hnmchallenge/models_prediction/dataset_logs/{dataset_name}")
        dir_path.mkdir(parents=True, exist_ok=True)

        log_filename = dir_path / f"{dataset_name}.log"

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
            f"Evaluating Ensemble:\n {self.RECS_NAME} \n\n cutoff:{self.cutoff} \n"
        )

        # retrieve recs
        recs = self.get_recommendations()
        self._check_recommendations_integrity(recs)

        # print how many recs per user we have on average
        logger.info(f"Average recs per user:{self.avg_recs_per_user}")

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
            left_on=[DEFAULT_USER_COL, "recs"],
            right_on=[DEFAULT_USER_COL, "article_id"],
            how="left",
        )

        pred = (
            merged[[DEFAULT_USER_COL, "recs", "rank"]]
            .copy()
            .rename(
                {f"recs": DEFAULT_ITEM_COL},
                axis=1,
            )
        )

        ground_truth = holdout_groundtruth[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].copy()
        logger.info(
            f"Remaining Users (at least one hit): {recs[DEFAULT_USER_COL].nunique()}"
        )
        logger.info("\nMetrics on ALL users")
        logger.info(f"MAP: {map_at_k(ground_truth, pred)}")
        logger.info(f"RECALL: {recall_at_k(ground_truth, pred)}")


if __name__ == "__main__":
    models = [
        # "cutf_100_PSGE_tw_True_rs_False_k_256",
        # "cutf_100_Popularity_cutoff_100",
        "cutf_100_ItemKNN_tw_True_rs_False",
        "cutf_100_TimePop_alpha_1.0",
        # "cutf_100_EASE_tw_True_rs_False_l2_0.001",
        # "cutf_40_Popularity_cutoff_40",
        # "cutf_0_BoughtItemsRecs",
    ]
    dataset = AILMLWDataset()
    for kind in ["full"]:  # , "full"]:
        ensemble = EnsembleRecs(
            models_list=models,
            kind=kind,
            dataset=dataset,
        )
        ensemble.save_recommendations(dataset_name="dataset_v1000")
        # ensemble.eval_recommendations(dataset_name="dataset_v03")
