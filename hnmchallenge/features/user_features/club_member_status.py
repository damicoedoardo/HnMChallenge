from unicodedata import name

import pandas as pd
from dotenv import main
from hnmchallenge.constant import DEFAULT_ITEM_COL, DEFAULT_USER_COL

from hnmchallenge.features.feature_interfaces import UserFeature


class ClubMemberStatus(UserFeature):
    FEATURE_NAME = "club_member_status_num"

    def __init__(self, dataset, kind: str) -> None:
        # we have the feature only for full
        kind = "full"
        super().__init__(dataset, kind)

    def _create_feature(self) -> pd.DataFrame:
        user_df = self.dataset.get_customers_df()

        cms = user_df[[DEFAULT_USER_COL, "club_member_status"]].copy()
        cms["club_member_status_num"] = 1
        cms.loc[cms["club_member_status"] == "PRE-CREATE", "club_member_status_num"] = 0
        cms = cms.drop("club_member_status", axis=1)
        print(cms)
        return cms
