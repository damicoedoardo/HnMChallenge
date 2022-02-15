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

        # check both customer id and item id are in str format
        assert isinstance(
            grp_recs_df.head()[DEFAULT_USER_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        assert isinstance(
            grp_recs_df.head()[DEFAULT_PREDICTION_COL].values[0], str
        ), f"Expected type str for col: {DEFAULT_USER_COL}"
        grp_recs_df.to_csv(
            str(self.dr.get_submission_folder() / sub_name) + ".csv", index=False
        )
        logger.info(set_color(f"Submission: {sub_name} created succesfully!", "yellow"))
