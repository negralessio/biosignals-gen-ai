import pandas as pd


class Preprocesser:

    def __init__(self, df_list: list[pd.DataFrame], condition: str, rolling_window_size: int, fixed_size: int):
        self.df_list: list[pd.DataFrame] = df_list
        self.condition: str = condition
        self.rolling_window_size: int = rolling_window_size
        self.fixed_size: int = fixed_size

        self.df_list_processed = None

    def preprocess_data(self):
        df_list_post = []
        for df in self.df_list:
            df = self.filter_condition_from_df(df, condition=self.condition)
            for col in self.get_eeg_cols(df):
                df = self.apply_rolling_window(df, feat=col, step=self.rolling_window_size)
            df = self.remove_nan_rows(df, col_to_inspect=f"EEG-L3-RW{self.rolling_window_size}")
            df = self.set_time_to_index(df)
            df = self.keep_only_relevant_features(df)
            df = self.cut_df_to_fixed_sized(df, desired_size=self.fixed_size)

            df_list_post.append(df)

        self.df_list_processed = df_list_post
        return df_list_post

    def filter_condition_from_df(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        df = df[df["Condition"] == condition]
        return df

    def apply_rolling_window(self, df: pd.DataFrame, feat: str, step: int = 250) -> pd.DataFrame:
        df = df.copy()
        # Apply rolling window with step size step on column feat
        df[f"{feat}-RW{step}"] = df[feat].rolling(window=step).mean()
        return df

    def remove_nan_rows(self, df: pd.DataFrame, col_to_inspect: str) -> pd.DataFrame:
        df = df[df[col_to_inspect].notna()]
        return df

    def get_eeg_cols(self, df: pd.DataFrame, search_str: str = "EEG") -> list[str]:
        return [x for x in list(df.columns) if x.startswith(search_str)]

    def set_time_to_index(self, df: pd.DataFrame, time_col="TS_UNIX") -> pd.DataFrame:
        try:
            df = df.set_index(time_col, drop=True)
        except KeyError:
            pass
        return df

    def keep_only_relevant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_keep = [x for x in list(df.columns) if x.endswith(f"RW{self.rolling_window_size}")]

        return df[cols_to_keep]

    def cut_df_to_fixed_sized(self, df: pd.DataFrame, desired_size: int = 10990,
                              remove_at_beginning: bool = False) -> pd.DataFrame:
        # Get current size
        N = df.shape[0]

        k = abs(N - desired_size)
        if remove_at_beginning:
            df = df.iloc[k:]
        else:
            df = df.iloc[:-k]

        return df
