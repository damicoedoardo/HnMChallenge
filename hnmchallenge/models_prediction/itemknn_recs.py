import logging

import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.models.itemknn.itemknn import ItemKNN
from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.stratified_dataset import StratifiedDataset
from hnmchallenge.utils.logger import set_color


class ItemKNNRecs(RecsInterface):
    def __init__(
        self,
        kind: str,
        dataset: StratifiedDataset,
        time_weight: bool = True,
        remove_seen: bool = False,
    ) -> None:
        super().__init__(kind, dataset)
        self.time_weight = time_weight
        self.remove_seen = remove_seen

        # set recommender name
        self.RECS_NAME = f"ItemKNN_tw_{time_weight}_rs_{remove_seen}"

    def get_recommendations(self, cutoff: int = 100) -> pd.DataFrame:
        data_df = (
            self.dataset.get_last_month_holdin()
            if self.kind == "train"
            else self.dr.get_filtered_full_data()
        )
        # data that you use to compute similarity
        # Using the full data available perform better
        data_sim = data_df[data_df["t_dat"] > "2020-08-31"]

        # instantiate the recommender algorithm
        recom = ItemKNN(self.dataset, time_weight=self.time_weight, topk=1000)

        print(set_color("Computing similarity...", "green"))
        recom.compute_similarity_matrix(data_df)
        recs = recom.recommend_multicore(
            interactions=data_df,
            batch_size=40_000,
            num_cpus=72,
            remove_seen=self.remove_seen,
            white_list_mb_item=None,
            cutoff=cutoff,
        )

        # recs_list = recs.groupby(DEFAULT_USER_COL)["article_id"].apply(list).to_frame()
        recs = recs.rename(
            {
                "article_id": f"{self.RECS_NAME}_recs",
                "prediction": f"{self.RECS_NAME}_score",
                "rank": f"{self.RECS_NAME}_rank",
            },
            axis=1,
        )
        return recs


if __name__ == "__main__":
    KIND = "train"
    TW = True
    REMOVE_SEEN = False
    dataset = StratifiedDataset()

    rec_ens = ItemKNNRecs(
        kind=KIND, time_weight=TW, remove_seen=REMOVE_SEEN, dataset=dataset
    )
    # rec_ens.eval_recommendations(cutoff=100)
    rec_ens.save_recommendations()
