import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.evaluation.python_evaluation import map_at_k, recall_at_k
from hnmchallenge.stratified_dataset import StratifiedDataset


def evaluate_recommender(recs: pd.DataFrame, dataset: StratifiedDataset):
    """Evaluate the recommendations over the groundtruth"""

    # load groundtruth and holdout data
    holdout_groundtruth = dataset.get_holdout_groundtruth()

    # merge recs and item groundtruth
    merged = pd.merge(
        recs,
        holdout_groundtruth,
        left_on=[DEFAULT_USER_COL, f"recs"],
        right_on=[DEFAULT_USER_COL, "article_id"],
        how="left",
    )

    # we have to remove the user for which we do not do at least one hit,
    # since we would not have the relavance for the items
    merged.loc[merged["article_id"].notnull(), "article_id"] = 1
    merged["hit_sum"] = merged.groupby(DEFAULT_USER_COL)["article_id"].transform("sum")

    merged_filtered = merged[merged["hit_sum"] > 0]

    pred = (
        merged[[DEFAULT_USER_COL, f"{self.RECS_NAME}_recs", f"{self.RECS_NAME}_rank"]]
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

    # coompute map and recall for whole the users
    map_k = map_at_k(ground_truth, pred)
    recall_k = recall_at_k(ground_truth, pred)
