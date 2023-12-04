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
    df_list = preprocesser.preprocess_data()

    # TODO:
    # Save preprocessed data so we do not have to execute preprocessing and loading everytime


if __name__ == '__main__':
    main()
