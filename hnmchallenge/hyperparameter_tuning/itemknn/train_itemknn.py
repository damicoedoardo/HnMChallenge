import argparse

from hnmchallenge.models.itemknn.itemknn import ItemKNN


if __name__ == "__main__":
    # initialise the dataset
    dataset = StratifiedDataset()
    data_df = dataset.get_last_day_holdin()

    parser = argparse.ArgumentParser("train itemknn")

    # Model parameters
    parser.add_argument("--time_weight", type=bool)
    parser.add_argument("--remove_seen", type=bool)

    parser.add_argument("--cutoff", type=int)

    # parse command line
    args = vars(parser.parse_args())

    recommender = ItemKNN(dataset, time_weight=args["time_weight"])
    recommender.compute_similarity_matrix(data_df)

    recs = recommender.recommend_multicore(
        interactions=data_df,
        batch_size=40_000,
        num_cpus=72,
        remove_seen=args["remove_seen"],
        white_list_mb_item=None,
        cutoff=args["cutoff"],
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
