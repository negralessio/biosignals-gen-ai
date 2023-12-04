import logging

import pandas as pd

import src.utils as utils

from src.dataloading import DataLoader
from src.preprocessing import Preprocesser

utils.setup_logging()
logger = logging.getLogger(__name__)


def main():
    cfg, config_path = utils.parse_config_file()

    # Load data as dataframes
    dataloader = DataLoader(**cfg["dataloading"])
    df_list: list[pd.DataFrame] = dataloader.load_data()

    # Apply preprocessing pipeline to each df in df_list
    preprocesser = Preprocesser(df_list=df_list, **cfg["preprocessing"])
    df_list: list[pd.DataFrame] = preprocesser.preprocess_data()

    # To save them
    # preprocesser.save_processed_dataframes(path_to_save="data/processed/P01/")


if __name__ == '__main__':
    main()
