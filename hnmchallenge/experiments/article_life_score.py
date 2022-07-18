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

if __name__ == "__main__":
    map_score = []
    recall_score = []
    dataset = AILMLWDataset()
    full_data = dataset.get_holdin()
    t_dat = full_data["t_dat"].max()
    dataset = AILMLWDataset()
    cut = [12, 50, 100, 200]
    for j in cut:
        x = []
        y = []
        for i in tqdm(range(100)):
            t_dat = full_data["t_dat"].max()
            t_dat1 = t_dat - pd.to_timedelta(i + 1, unit="D")
            candidate_items = full_data[full_data["t_dat"] >= t_dat1][
                ["article_id"]
            ].drop_duplicates()
            candidate_items = candidate_items.values.squeeze()
            TW = True
            REMOVE_SEEN = False
            FC = True
            dataset = AILMLWDataset()
            for kind in ["train"]:  # , "full"]:        # for kind in ["train", "full"]:
                rec_ens = ItemKNNRecs(
                    kind=kind,
                    cutoff=j,
                    time_weight=TW,
                    remove_seen=REMOVE_SEEN,
                    dataset=dataset,
                    filter_on_candidates=candidate_items,
                )
                score, recall = rec_ens.eval_recommendations()
                x.append(score)
                y.append(recall)
        map_score.append(x)
        recall_score.append(y)

    DATASET_NAME = "AILMLW_dataset"
    _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))
    _DATASET_PATH = _DATA_PATH / "datasets" / DATASET_NAME
    _HOLDIN_PATH = Path(_DATASET_PATH / Path("map_score.feather"))
    df = {
        "MAP@12": map_score[0],
        "recall@12": recall_score[0],
        "MAP@50": map_score[1],
        "recall@50": recall_score[1],
        "MAP@100": map_score[2],
        "recall@100": recall_score[2],
        "MAP@200": map_score[3],
        "recall@200": recall_score[3],
    }
    article_score = pd.DataFrame(df)
    article_score.to_feather(_HOLDIN_PATH)
