import logging

import pandas as pd

from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader
from hnmchallenge.utils.logger import set_color

logger = logging.getLogger(__name__)


class SubmissionHandler:
    def __init__(self):
        self.dr = DataReader()

    def create_submission(self, recs_df: pd.DataFrame, sub_name: str) -> None:
        assert DEFAULT_USER_COL in recs_df.columns, f"Missing col: {DEFAULT_USER_COL}"
        assert (
            DEFAULT_PREDICTION_COL in recs_df.columns
        ), f"Missing col: {DEFAULT_PREDICTION_COL}"

        user_map_dict, item_map_dict = self.dr.get_new_raw_mapping_dict()
        grp_recs_df = recs_df.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(list)
        grp_recs_df = grp_recs_df.to_frame().reset_index()
        # map back to original ids
        grp_recs_df[DEFAULT_USER_COL] = grp_recs_df[DEFAULT_USER_COL].apply(
            lambda x: user_map_dict.get(x)
        )
        grp_recs_df[DEFAULT_ITEM_COL] = grp_recs_df[DEFAULT_ITEM_COL].apply(
            lambda x: " ".join(list(map(item_map_dict.get, x)))
        )
        grp_recs_df = grp_recs_df.rename(
            columns={DEFAULT_ITEM_COL: DEFAULT_PREDICTION_COL}
        )
        # concat the recommendations for zero length users with the other predictions
        zero_interactions_recs = self.dr.get_zero_interactions_recs()
        final_recs = pd.concat([grp_recs_df, zero_interactions_recs], axis=0)

        # check both customer id and item id are in str format
        assert isinstance(
            final_recs.head()[DEFAULT_USER_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        assert isinstance(
            final_recs.head()[DEFAULT_PREDICTION_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        final_recs.to_csv(
            str(self.dr.get_submission_folder() / sub_name) + ".csv", index=False
        )
        logger.info(set_color(f"Submission: {sub_name} created succesfully!", "yellow"))

    def create_submission_filtered_data(
        self, recs_dfs: list[pd.DataFrame], sub_name: str
    ) -> None:
        assert len(recs_dfs) > 0, "recs_df is empty"
        for df in recs_dfs:
            assert DEFAULT_USER_COL in df.columns, f"Missing col: {DEFAULT_USER_COL}"
            # assert (
            #     DEFAULT_PREDICTION_COL in df.columns
            # ), f"Missing col: {DEFAULT_PREDICTION_COL}"

        user_map_dict, item_map_dict = self.dr.get_filtered_new_raw_mapping_dict()
        # concatemate together the predictions for the different user clusters
        recs_dfs_concat = pd.concat(recs_dfs, axis=0)

        grp_recs_df = recs_dfs_concat.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(
            list
        )
        grp_recs_df = grp_recs_df.to_frame().reset_index()
        # map back to original ids
        grp_recs_df[DEFAULT_USER_COL] = grp_recs_df[DEFAULT_USER_COL].apply(
            lambda x: user_map_dict.get(x)
        )
        grp_recs_df[DEFAULT_ITEM_COL] = grp_recs_df[DEFAULT_ITEM_COL].apply(
            lambda x: " ".join(list(map(item_map_dict.get, x)))
        )
        grp_recs_df = grp_recs_df.rename(
            columns={DEFAULT_ITEM_COL: DEFAULT_PREDICTION_COL}
        )
        # concat the recommendations for zero length users with the other predictions
        zero_interactions_recs = self.dr.get_filtered_zero_interactions_recs()
        final_recs = pd.concat([grp_recs_df, zero_interactions_recs], axis=0)

        # check both customer id and item id are in str format
        assert isinstance(
            final_recs.head()[DEFAULT_USER_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        assert isinstance(
            final_recs.head()[DEFAULT_PREDICTION_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        final_recs.to_csv(
            str(self.dr.get_submission_folder() / sub_name) + ".csv", index=False
        )
        logger.info(
            set_color(
                f"Submission with Filtered Data: {sub_name} created succesfully!",
                "yellow",
            )
        )

    def create_submission_filtered_data_full_users(
        self, recs_dfs: list[pd.DataFrame], sub_name: str
    ) -> None:
        assert len(recs_dfs) > 0, "recs_df is empty"
        for df in recs_dfs:
            assert DEFAULT_USER_COL in df.columns, f"Missing col: {DEFAULT_USER_COL}"
            # assert (
            #     DEFAULT_PREDICTION_COL in df.columns
            # ), f"Missing col: {DEFAULT_PREDICTION_COL}"

        user_map_dict, item_map_dict = self.dr.get_filtered_new_raw_mapping_dict()
        # concatemate together the predictions for the different user clusters
        recs_dfs_concat = pd.concat(recs_dfs, axis=0)

        grp_recs_df = recs_dfs_concat.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL].apply(
            list
        )
        grp_recs_df = grp_recs_df.to_frame().reset_index()
        # map back to original ids
        grp_recs_df[DEFAULT_USER_COL] = grp_recs_df[DEFAULT_USER_COL].apply(
            lambda x: user_map_dict.get(x)
        )
        grp_recs_df[DEFAULT_ITEM_COL] = grp_recs_df[DEFAULT_ITEM_COL].apply(
            lambda x: " ".join(list(map(item_map_dict.get, x)))
        )
        grp_recs_df = grp_recs_df.rename(
            columns={DEFAULT_ITEM_COL: DEFAULT_PREDICTION_COL}
        )

        final_recs = grp_recs_df

        # check both customer id and item id are in str format
        assert isinstance(
            final_recs.head()[DEFAULT_USER_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        assert isinstance(
            final_recs.head()[DEFAULT_PREDICTION_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        final_recs.to_csv(
            str(self.dr.get_submission_folder() / sub_name) + ".csv", index=False
        )
        logger.info(
            set_color(
                f"Submission with Filtered Data: {sub_name} created succesfully!",
                "yellow",
            )
        )
