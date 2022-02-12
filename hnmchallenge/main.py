from unittest import removeResult
from dotenv import load_dotenv
import os
import pandas as pd
from hnmchallenge.data_reader import DataReader


if __name__ == "__main__":

    print(os.environ.get("DATA_PATH"))
    data_reader = DataReader()
    print(data_reader.get_articles())
    print(data_reader.get_customer())
    print(data_reader.get_transactions())
