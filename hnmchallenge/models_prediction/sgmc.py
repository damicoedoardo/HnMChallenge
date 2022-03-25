import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.dataset import Dataset
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.filtered_dataset import FilterdDataset
from hnmchallenge.models.ease.ease import EASE
from hnmchallenge.models.sgmc.sgmc import SGMC
from hnmchallenge.models.top_pop import TopPop
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color

# MODEL PARAMETERS
TIME_WEIGHT = True
K = 128

KIND = "train"
CUTOFF = 100
assert KIND in ["train", "full"], "kind should be train or full"
RECS_NAME = f"ease_{CUTOFF}_tw_{TIME_WEIGHT}.feather"

# SAVE THE PREDICTION OR EVAL THE MODEL
SAVE_PREDICTIONS = False
EVAL = not SAVE_PREDICTIONS

if __name__ == "__main__":
    if KIND == "full":
        # the only thing we can do when kind is full is to save the predictions
        assert (
            SAVE_PREDICTIONS
        ), "kind is full and eval is True, not possible to evaluate on full predictions..."

    dataset = StratifiedDataset()
    dr = DataReader()

    recom = SGMC(dataset, time_weight=True, k=K)

    holdin = dataset.get_last_month_holdin()
    fd = dr.get_filtered_full_data()

    # set the correct data to use
    data_df = None
    if KIND == "train":
        data_df = holdin
        data_sim = holdin[holdin["t_dat"] > "2020-08-31"]
        data_sim = holdin
    else:
        data_df = fd
        data_sim = fd[fd["t_dat"] > "2020-08-31"]

    print(set_color("Computing similarity...", "green"))
    recom.compute_similarity_matrix(data_sim)
    recs = recom.recommend_multicore(
        interactions=data_df,
        batch_size=60_000,
        num_cpus=72,
        remove_seen=False,
        white_list_mb_item=None,
        cutoff=CUTOFF,
    )

    # recs_list = recs.groupby(DEFAULT_USER_COL)["article_id"].apply(list).to_frame()
    recs = recs.rename(
        {
            "article_id": "recs",
            "prediction": f"{recom.name}_score",
            "rank": f"{recom.name}_rank",
        },
        axis=1,
    )

    if SAVE_PREDICTIONS:
        save_name = f"{KIND}_{RECS_NAME}"
        recs.reset_index(drop=True).to_feather(
            dr.get_preprocessed_data_path() / save_name
        )

    if EVAL:
        ##################################################
        # COMPUTE RECOMMENDATION METRICS
        ##################################################

        # retrieve the holdout
        holdout = dataset.get_last_month_holdout()
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

        pred = (
            merged[[DEFAULT_USER_COL, "recs", f"{recom.name}_rank"]]
            .copy()
            .rename({"recs": DEFAULT_ITEM_COL, f"{recom.name}_rank": "rank"}, axis=1)
        )
        pred_filtered = (
            merged_filtered[[DEFAULT_USER_COL, "recs", f"{recom.name}_rank"]]
            .copy()
            .rename({"recs": DEFAULT_ITEM_COL, f"{recom.name}_rank": "rank"}, axis=1)
        )
        ground_truth = holdout[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].copy()

        print(
            f"Remaining Users (at least one hit): {merged_filtered[DEFAULT_USER_COL].nunique()}"
        )

        print(set_color(f"RECOMMENDER: {recom.name}", "red"))

        print(set_color("\nMetrics on ALL users", "cyan"))
        print(set_color(f"MAP@{CUTOFF}: {map_at_k(ground_truth, pred)}", "cyan"))
        print(set_color(f"RECALL@{CUTOFF}: {recall_at_k(ground_truth, pred)}", "cyan"))

        print(set_color("\nMetrics on ONE-HIT users", "yellow"))
        print(
            set_color(
                f"MAP@{CUTOFF}: {map_at_k(ground_truth, pred_filtered)}", "yellow"
            )
        )
        print(
            set_color(
                f"RECALL@{CUTOFF}: {recall_at_k(ground_truth, pred_filtered)}",
                "yellow",
            )
        )
