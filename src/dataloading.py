""" Class that handles the data loading """
import os
import pandas as pd
import logging
import warnings

import src.utils as utils

utils.setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class DataLoader:
    def __init__(self, path_to_data: str):
        """
        :param path_to_data: str -- Path to where the session data lies (see config.yaml)
        """
        self.path_to_data: str = path_to_data

        self.data_paths = None
        self.data = None

    def load_data(self) -> list[pd.DataFrame]:
        """
        First gets the path to each individual .csv file and then loads them as dataframes

        :return: data: list[pd.DataFrame]
        """
        _ = self.get_paths_to_data()
        logger.info(f"Found {len(self.data_paths)} .csv files in input path '{self.path_to_data}' ...")

        _ = self.load_data_as_dataframe()
        logger.info(f"Parsed {len(self.data)} .csv files into DataFrames ...")

        return self.data

    def get_paths_to_data(self, append_path: bool = True) -> list[str]:
        """ Returns the paths to the .csv files

        :param append_path: Whether to attend the relative path (from root) to the .csv file
        :return: data_paths: list[str]
        """
        # Get list of csv files
        data_paths = os.listdir(self.path_to_data)

        # Keep only csv data
        data_paths = [x for x in data_paths if x.endswith(".csv")]

        # Append path from current dir to csv file if input param is True
        if append_path:
            data_paths = [os.path.join(self.path_to_data, x) for x in data_paths]

        data_paths.sort()
        self.data_paths = data_paths
        logger.debug(f"Identified .csv files:\n{self.data_paths}")
        return data_paths

    def load_data_as_dataframe(self) -> list[pd.DataFrame]:
        """ Loads the .csv file into dataframes

        :return: list[pd.DataFrames]
        """
        data = []
        for path in self.data_paths:
            df = pd.read_csv(path)
            # Convert 'TS_UNIX' to datetime
            df["TS_UNIX"] = pd.to_datetime(df["TS_UNIX"], infer_datetime_format=True)  # format="%Y-%m-%d %H:%M:%S.%f"
            data.append(df)

        self.data = data
        return data
