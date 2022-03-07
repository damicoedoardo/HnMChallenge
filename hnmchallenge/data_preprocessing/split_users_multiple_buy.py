from unittest.main import main
import numpy as np
import pandas as pd
from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader

if __name__ == "__main__":
    dr = DataReader()
    transaction = dr.get_filtered_full_data()
    item_per_user1 = transaction.groupby("customer_id")["article_id"].apply(list)
    unique_item_per_user1 = item_per_user1.apply(np.unique)
    df_multiple1 = item_per_user1.to_frame()
    df_unique1 = unique_item_per_user1.to_frame()
    df_multiple1["count"] = df_multiple1.apply(lambda row: len(row["article_id"]), axis=1)
    df_unique1["count"] = df_unique1.apply(lambda row: len(row["article_id"]), axis=1)
    merge_df1 = pd.merge(df_multiple1, df_unique1, on="customer_id")
    merge_df1["diff"] = 1 - (merge_df1["count_y"]/merge_df1["count_x"])
    user_diff=merge_df1.drop(["article_id_x","count_x","article_id_y","count_y"], axis=1)
    user_diff.reset_index(level=0, inplace=True)
    user_diff.to_feather(dr.get_preprocessed_data_path() / "filtered_split_user_multiple_buy.feather")