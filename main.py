import src.utils as utils

from src.dataloading import DataLoader
from src.preprocessing import Preprocesser


def main():
    cfg, config_path = utils.parse_config_file()

    # Load data as dataframes
    dataloader = DataLoader(**cfg["dataloading"])
    df_list = dataloader.load_data()

    # Apply preprocessing pipeline to each df in df_list
    preprocesser = Preprocesser(df_list=df_list, **cfg["preprocessing"])
    df_list = preprocesser.preprocess_data()



if __name__ == '__main__':
    main()
