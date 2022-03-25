from unittest.main import main

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader

if __name__ == "__main__":
    dr = DataReader()
    zero_int_users = dr.get_zero_interatction_users()
    transaction = dr.get_full_data()
    pop_items = (
        transaction[transaction["t_dat"] > "2020-08-31"]
        .groupby("article_id")
        .count()
        .sort_values("t_dat", ascending=False)
        .iloc[0:12]
        .index.values
    )
    # map items to raw ids
    _, item_map_dict = dr.get_new_raw_mapping_dict()
    pop_items_raw = [item_map_dict[i] for i in pop_items]
    pop_items_raw_str = " ".join(pop_items_raw)
    zero_int_users[DEFAULT_PREDICTION_COL] = pop_items_raw_str
    zero_int_users.to_feather(
        dr.get_preprocessed_data_path() / "zero_interactions_recs.feather"
    )
