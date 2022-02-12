from hnmchallenge.data_reader import DataReader
from hnmchallenge.constant import *

if __name__ == "__main__":
    dr = DataReader()
    transaction = dr.get_full_data()
    tu = dr.get_target_user()
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
    tu[DEFAULT_PRED_COL] = pop_items_raw_str
    dr.create_submission(tu, sub_name="TopPop-1M")
