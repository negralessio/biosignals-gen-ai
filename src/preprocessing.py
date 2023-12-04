""" Class that handles the preprocessing of the input dataframes """
import pandas as pd
import os
import logging
import time

import src.utils as utils

utils.setup_logging()
logger = logging.getLogger(__name__)


class Preprocesser:

    def __init__(self, df_list: list[pd.DataFrame], condition: str, rolling_window_size: int, fixed_size: int):
        """
        :param df_list: list[pd.DataFrame] -- Input Dataframes stored in a list
        :param condition: str -- Current condition to analyse
        :param rolling_window_size: int -- Rolling window size to aggregate rows
        :param fixed_size: int -- Which fixed size of the time series to be desired
        """
        self.df_list: list[pd.DataFrame] = df_list
        self.condition: str = condition
        self.rolling_window_size: int = rolling_window_size
        self.fixed_size: int = fixed_size

        self.df_list_processed = None

    def preprocess_data(self) -> list[pd.DataFrame]:
        """ Applies a number of preprocessing steps to the input dataframes

        :return: df_list_processed: list[pd.DataFrame]
        """
        logger.info(f"Starting preprocessing pipeline (Condition: {self.condition},"
                    f" Window Size: {self.rolling_window_size},"
                    f" Fixed Size: {self.fixed_size}) ...")
        start_time = time.time()
        df_list_post = []

        # Iterate through each provided dataframe and apply multiple preprocessing steps
        for df in self.df_list:
            df = self.filter_condition_from_df(df, condition=self.condition)
            for col in self.get_eeg_cols(df):
                df = self.apply_rolling_window(df, feat=col, step=self.rolling_window_size)
            df = self.remove_nan_rows(df, col_to_inspect=f"EEG-L3-RW{self.rolling_window_size}")
            df = self.set_time_to_index(df)
            df = self.keep_only_relevant_features(df)
            df = self.cut_df_to_fixed_sized(df, desired_size=self.fixed_size)
            # Append preprocessed dataframe to list
            df_list_post.append(df)

        dur = time.time() - start_time
        logger.info(f"Finished preprocessing pipeline (Duration: {dur:.2f}s) ...")

        self.df_list_processed = df_list_post
        return df_list_post

    def filter_condition_from_df(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """ Selects the corresponding rows from the given dataframe based on the condition """
        df = df[df["Condition"] == condition]
        return df

    def apply_rolling_window(self, df: pd.DataFrame, feat: str, step: int = 250) -> pd.DataFrame:
        """ Applies rolling window with mean() operation and step size step on the df """
        df = df.copy()
        # Apply rolling window with step size step on column feat
        df[f"{feat}-RW{step}"] = df[feat].rolling(window=step).mean()
        return df

    def remove_nan_rows(self, df: pd.DataFrame, col_to_inspect: str) -> pd.DataFrame:
        """ Remove NaN values due to the rolling window """
        df = df[df[col_to_inspect].notna()]
        return df

    def get_eeg_cols(self, df: pd.DataFrame, search_str: str = "EEG") -> list[str]:
        """ Returns the features columns """
        return [x for x in list(df.columns) if x.startswith(search_str)]

    def set_time_to_index(self, df: pd.DataFrame, time_col="TS_UNIX") -> pd.DataFrame:
        """ Sets the time column as index """
        try:
            df = df.set_index(time_col, drop=True)
        except KeyError:
            pass
        return df

    def keep_only_relevant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Removes non-features of the input dataframe df """
        cols_to_keep = [x for x in list(df.columns) if x.endswith(f"RW{self.rolling_window_size}")]

        return df[cols_to_keep]

    def cut_df_to_fixed_sized(self, df: pd.DataFrame, desired_size: int = 10990,
                              remove_at_beginning: bool = False) -> pd.DataFrame:
        """ Sets the time series to a fixed size """
        # Get current size
        N = df.shape[0]

        # Determine how many samples to cut off
        k = abs(N - desired_size)
        if remove_at_beginning:
            df = df.iloc[k:]
        else:
            df = df.iloc[:-k]

        return df

    def save_processed_dataframes(self, path_to_save: str = "data/processed/P01/") -> None:
        """ Saves processed dataframes into the specified directory

        :param path_to_save: str -- Directory to save processed dfs
        :return: None
        """
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        for i, df in enumerate(self.df_list_processed):
            df.to_csv(path_to_save + f"{i+1}.csv")

        logger.info(f"Saved data in '{path_to_save}' ...")
