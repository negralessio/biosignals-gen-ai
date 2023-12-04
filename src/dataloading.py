""" Class that handles the data loading """
import os
import pandas as pd


class DataLoader:
    def __init__(self, path_to_data):
        self.path_to_data: str = path_to_data

        self.data_paths = None
        self.data = None

    def load_data(self) -> list[pd.DataFrame]:
        _ = self.get_paths_to_data()
        _ = self.load_data_as_dataframe()
        return self.data

    def get_paths_to_data(self, append_path: bool = True) -> list[str]:
        # Get list of csv files
        data_paths = os.listdir(self.path_to_data)

        # Keep only csv data
        data_paths = [x for x in data_paths if x.endswith(".csv")]

        # Append path from current dir to csv file if input param is True
        if append_path:
            data_paths = [os.path.join(self.path_to_data, x) for x in data_paths]

        data_paths.sort()
        self.data_paths = data_paths
        return data_paths

    def load_data_as_dataframe(self) -> list[pd.DataFrame]:
        data = []
        for path in self.data_paths:
            df = pd.read_csv(path)
            # Convert 'TS_UNIX' to datetime
            df["TS_UNIX"] = pd.to_datetime(df["TS_UNIX"], format="%Y-%m-%d %H:%M:%S.%f")
            data.append(df)

        self.data = data
        return data
