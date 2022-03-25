from hnmchallenge.constant import *
from hnmchallenge.data_reader import DataReader


def save_target_user() -> None:
    """Save target users in feather format"""
    dr = DataReader()
    ss = dr.get_sample_submission()
    target_user = ss[[DEFAULT_USER_COL]]
    target_user.to_feather(dr.get_preprocessed_data_path() / "target_user.feather")


if __name__ == "__main__":
    save_target_user()
