import logging

import pandas as pd
import numpy as np

import src.utils as utils

from src.dataloading import DataLoader
from src.preprocessing import Preprocesser
from src.modelling import VAE

utils.setup_logging()
logger = logging.getLogger(__name__)


def main():
    cfg, config_path = utils.parse_config_file()

    # Load data as dataframes
    dataloader = DataLoader(**cfg["dataloading"])
    df_list: list[pd.DataFrame] = dataloader.load_data()

    # Apply preprocessing pipeline to each df in df_list
    preprocesser = Preprocesser(df_list=df_list, **cfg["preprocessing"])
    tensor: np.array = preprocesser.preprocess_data()

    # To save preprocessed dataframes
    # preprocesser.save_processed_dataframes(path_to_save="data/processed/P01/")

    # Create VAE object, compile and fit the network
    vae = VAE(tensor=tensor, **cfg["architecture"])
    vae.compile(optimizer="adam")
    vae.build((None, tensor.shape[1], tensor.shape[2]))
    vae.fit(tensor, epochs=32, batch_size=2)     # Or use **cfg["fitting"]

    # TODO: Train Test Split, modelling Module improvement, ...


if __name__ == '__main__':
    main()
