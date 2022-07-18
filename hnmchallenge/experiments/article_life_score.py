import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.datasets.all_items_last_month_last_week import AILMLWDataset
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.models_prediction.itemknn_recs import ItemKNNRecs
from hnmchallenge.utils.logger import set_color
from pathlib import Path
import os

# CUT = [12, 50, 100, 200]
CUT = [12]
TW = False
REMOVE_SEEN = True
FC = True
KIND = "train"

if __name__ == "__main__":
    map_score = []
    recall_score = []

    dataset = AILMLWDataset()
    holdin = dataset.get_holdin()

    max_holdin_date = holdin["t_dat"].max()

    for cutoff in CUT:
        recall_list = []
        map_list = []
        for i in tqdm(range(100)):

            current_date = max_holdin_date - pd.to_timedelta(i + 1, unit="D")
            print(current_date)

            candidate_items = holdin[holdin["t_dat"] > current_date][
                ["article_id"]
            ].drop_duplicates()

            candidate_items = candidate_items.values.squeeze()
            print(f"Candidate items {i+1} days back in time: {len(candidate_items)}")

            model = ItemKNNRecs(
                kind=KIND,
                cutoff=cutoff,
                time_weight=TW,
                remove_seen=REMOVE_SEEN,
                dataset=dataset,
                filter_on_candidates=candidate_items,
            )
            map_res, recall_res = model.eval_recommendations()
            map_list.append(map_res)
            recall_list.append(recall_res)

        map_score.append(map_list)
        recall_score.append(recall_list)

    # DATASET_NAME = "AILMLW_dataset"
    # _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))
    # _DATASET_PATH = _DATA_PATH / "datasets" / DATASET_NAME
    # _HOLDIN_PATH = Path(_DATASET_PATH / Path("map_score.feather"))
    # df = {
    #     "MAP@12": map_score[0],
    #     "recall@12": recall_score[0],
    #     "MAP@50": map_score[1],
    #     "recall@50": recall_score[1],
    #     "MAP@100": map_score[2],
    #     "recall@100": recall_score[2],
    #     "MAP@200": map_score[3],
    #     "recall@200": recall_score[3],
    # }
    # article_score = pd.DataFrame(df)
    # article_score.to_feather(_HOLDIN_PATH)
