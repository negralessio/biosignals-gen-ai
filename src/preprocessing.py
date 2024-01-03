""" Class that handles the preprocessing of the input dataframes """
import os
import logging
import time
import pandas as pd
import numpy as np

import src.utils as utils

from sklearn.preprocessing import MinMaxScaler

utils.setup_logging()
logger = logging.getLogger(__name__)


class Preprocesser:

    def __init__(self, df_list: list[pd.DataFrame], condition: str, partition_size: int, fixed_size: int):
        """
        :param df_list: list[pd.DataFrame] -- Input Dataframes stored in a list
        :param condition: str -- Current condition to analyse
        :param partition_size: int -- Number of rows to take as one chunk
        :param fixed_size: int -- Which fixed size of the time series to be desired
        """
        self.df_list: list[pd.DataFrame] = df_list
        self.condition: str = condition
        self.partition_size: int = partition_size
        self.fixed_size: int = fixed_size

        # List of processed DataFrames after preprocess_data() func
        self.df_list_processed = None
        # Stacked 3D Numpy Array of preprocessed data after preprocess_data() func
        self.tensor = None
        # Scaled MinMax or StandardScaler Object after scale_df() func
        self.scaler = None

    def preprocess_data(self) -> np.array:
        """ Applies a number of preprocessing steps to the input dataframes

        :return: tensor: np.array -- 3D Tensor of processed data
        """
        logger.info(f"Starting preprocessing pipeline (Condition: {self.condition},"
                    f" Partition Size: {self.partition_size},"
                    f" Fixed Size: {self.fixed_size}) ...")
        start_time = time.time()
        df_list_post = []

        # Iterate through each provided dataframe and apply multiple preprocessing steps
        for df in self.df_list:
            df = self.filter_condition_from_df(df, condition=self.condition)
            df = self.remove_nan_rows(df, col_to_inspect=f"EEG-L3")
            df = self.set_time_to_index(df)
            df = self.keep_only_relevant_features(df)
            df = self.cut_df_to_fixed_sized(df, desired_size=self.fixed_size)
            df = self.scale_df(df)
            df_chunks: list[pd.DataFrame] = self.get_chunks_of_size_n(df, n=self.partition_size)

            # Append processed chunks in df_list_post
            df_list_post = df_list_post + df_chunks

        self.df_list_processed = df_list_post
        _ = self.stack_to_3d()

        dur = time.time() - start_time
        logger.info(f"Finished preprocessing pipeline (Duration: {dur:.2f}s) ...")

        return self.tensor

    def filter_condition_from_df(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """ Selects the corresponding rows from the given dataframe based on the condition """
        df = df[df["Condition"] == condition]
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
        cols_to_keep = self.get_eeg_cols(df, search_str="EEG")

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

    def scale_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Applies MinMaxScaler on the input df

        :param df: pd.DataFrame -- Input data to be scaled
        :return: df_scaled: pd.DataFrame
        """
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        self.scaler = scaler
        return df_scaled

    def get_chunks_of_size_n(self, df: pd.DataFrame, n: int = 250) -> list[pd.DataFrame]:
        """ Partitions given input df into chunks of size n

        :param df: pd.DataFrame
        :param n: int -- Size of the chunks
        :return: list[pd.DataFrame]
        """
        return [df[i:i+n] for i in range(0, len(df), n)]

    def stack_to_3d(self) -> np.array:
        """ Stacks the data in df_list_processed as 3D tensor ready for LSTM input

        :return: np.array -- 3D Tensor of (Samples, Sequence Length, N_Features)
        """
        logger.info("Stacking data to 3D tensor ...")
        sequence_length = self.df_list_processed[0].shape[0]

        # Convert DataFrames to numpy arrays and reshape them
        reshaped_data = []
        for df in self.df_list_processed:
            # Convert DataFrame to numpy array
            data_array = df.values  # Assuming your DataFrame is named 'df'

            # Calculate the number of sequences that can be extracted
            num_sequences = len(data_array) - sequence_length + 1

            # Extract sequences of length 'sequence_length'
            for i in range(num_sequences):
                sequence = data_array[i:i + sequence_length]
                reshaped_data.append(sequence)

        # Convert reshaped_data to a numpy array
        reshaped_data = np.array(reshaped_data)
        # Save as attribute and return results
        self.tensor = reshaped_data
        return reshaped_data

    def save_processed_dataframes(self, path_to_save: str = "data/processed/P01/") -> None:
        """ Saves processed dataframes into the specified directory

        :param path_to_save: str -- Directory to save processed dfs
        :return: None
        """
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        for i, df in enumerate(self.df_list_processed):
            df.to_csv(path_to_save + f"{i + 1}.csv")

        logger.info(f"Saved data in '{path_to_save}' ...")
