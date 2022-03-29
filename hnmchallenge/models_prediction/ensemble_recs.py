import logging
from datetime import datetime
from functools import reduce
from pathlib import Path
from pickle import FALSE

import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.models_prediction.bought_items_recs import BoughtItemsRecs
from hnmchallenge.models_prediction.ease_recs import EaseRecs
from hnmchallenge.models_prediction.itemknn_recs import ItemKNNRecs
from hnmchallenge.models_prediction.popularity_recs import PopularityRecs
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.models_prediction.time_pop import TimePop
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color
from matplotlib.pyplot import axis


class EnsembleRecs(RecsInterface):
    RECS_NAME = None

    def __init__(
        self, models_list: list, kind: str, dataset: StratifiedDataset
    ) -> None:
        # Check that all the models passed are extending Recs Interface
        assert (
            len(models_list) > 1
        ), "At least 2 models should be passed to create an Ensemble !"
        all(
            [issubclass(m.__class__, RecsInterface) for m in models_list]
        ), "Not all the models passed are extending `RecsInterface`"

        cutoffs_list = [m.cutoff for m in models_list]
        super().__init__(kind=kind, dataset=dataset, cutoff=cutoffs_list)

        # we have to check that the kind of the ensamble correspond with the kind of each model passed
        assert all(
            [model.kind == kind for model in models_list]
        ), f"Ensemble kind:{kind}, not all the model passed has the same kind of the ensamble!"

        self.models_list = models_list
        self.RECS_NAME = self._create_ensemble_name()
        self.avg_recs_per_user = None

    def _create_ensemble_name(self):
        """Create a meaningfull name for the ensamble model"""
        models_names = [model.RECS_NAME for model in self.models_list]
        ensemble_name = "\n".join(models_names)
        print(f"Creating ensemble with:\n{ensemble_name}\n\n")
        return ensemble_name

    def get_recommendations(self) -> pd.DataFrame:
        # retrieve the recommendations from the different models and join them `outer`
        def _merge_dfs(tup_x, tup_y):
            recs_x = tup_x[1]
            recs_y = tup_y[1]

            name_x = tup_x[0]
            name_y = tup_y[0]

            # change the names for the different recommendation algs
            if name_x != "recs":
                name_x = name_x + "_recs"

            if name_y != "recs":
                name_y = name_y + "_recs"

            merged = pd.merge(
                recs_x,
                recs_y,
                left_on=[DEFAULT_USER_COL, name_x],
                right_on=[DEFAULT_USER_COL, name_y],
                how="outer",
            )
            merged["recs"] = merged.filter(like="recs").ffill(axis=1).iloc[:, -1]

            cols_to_drop = [name for name in [name_x, name_y] if name != "recs"]
            merged = merged.drop(cols_to_drop, axis=1)
            return ("recs", merged)

        # recs_list = [(model_name, recs_df), ...]
        recs_tup_list = [
            (model.RECS_NAME, model.get_recommendations()) for model in self.models_list
        ]
        merged_recs_df = reduce(_merge_dfs, recs_tup_list)[1]

        # store average recs per user
        self.avg_recs_per_user = (
            merged_recs_df.groupby(DEFAULT_USER_COL).size().values.mean()
        )
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
                left_on=[DEFAULT_USER_COL, "recs"],
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

        col_to_drop = [col for col in recs.columns if "rank" in col]
        print(col_to_drop)
        recs = recs.drop(col_to_drop, axis=1)
        print(recs.columns)

        # save the retrieved recommendations
        save_name = f"{dataset_name}.feather"
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

        # load groundtruth and holdout data
        holdout_groundtruth = self.dataset.get_holdout_groundtruth()
        holdout = self.dataset.get_last_day_holdout()

        # merge recs and item groundtruth
        merged = pd.merge(
            recs,
            holdout_groundtruth,
            left_on=[DEFAULT_USER_COL, "recs"],
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

        # TODO to be changed ! rank is no more valid
        pred = (
            merged[[DEFAULT_USER_COL, "recs", "rank"]]
            .copy()
            .rename(
                {f"recs": DEFAULT_ITEM_COL},
                axis=1,
            )
        )
        pred_filtered = (
            merged_filtered[[DEFAULT_USER_COL, "recs", "rank"]]
            .copy()
            .rename(
                {"recs": DEFAULT_ITEM_COL},
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


if __name__ == "__main__":
    KIND = "train"

    dataset = StratifiedDataset()

    rec_ens_1 = ItemKNNRecs(
        kind=KIND, cutoff=100, time_weight=True, remove_seen=True, dataset=dataset
    )
    # rec_ens_1 = ItemKNNRecs(
    #     kind=KIND, cutoff=100, time_weight=False, remove_seen=False, dataset=dataset
    # )
    # rec_ens_2 = EaseRecs(
    #     kind=KIND, cutoff=100, dataset=dataset, l2=0.1, remove_seen=True, time_weight=False
    # )
    # rec_ens_2 = PopularityRecs(kind=KIND, cutoff=20, dataset=dataset)

    rec_ens_2 = BoughtItemsRecs(kind=KIND, dataset=dataset)

    ensemble = EnsembleRecs(
        models_list=[rec_ens_1, rec_ens_2], kind=KIND, dataset=dataset
    )
    ensemble.save_recommendations(dataset_name="dataset_v13")
    # ensemble.eval_recommendations(dataset_name="dataset_v13")
