import logging

import pandas as pd
import numpy as np

import src.utils as utils
import src.modelling as modelling

from src.dataloading import DataLoader
from src.preprocessing import Preprocesser
from src.vae_dense import DenseVAE

from sklearn.model_selection import train_test_split

from typing import Union

utils.setup_logging()
logger = logging.getLogger(__name__)


def main():
    cfg, config_path = utils.parse_config_file()
    MODEL_ID: str = utils.get_model_id(n_epochs=cfg["fitting"]["epochs"], n_batches=cfg["fitting"]["batch_size"],
                                       val_split=0.2,
                                       latent_dim=cfg["architecture"]["latent_dims"])
    # Load data as dataframes
    dataloader = DataLoader(**cfg["dataloading"])
    df_list: list[pd.DataFrame] = dataloader.load_data()

    # Apply preprocessing pipeline to each df in df_list and each condition
    pp_data, tensor, y_tensor, conditions_df = _stack_all_conditions_data(df_list=df_list, **cfg["preprocessing"])
    # Process labels
    y_label, classes = _process_labels(y_tensor)

    # Train Test Split
    train_ind, test_ind = train_test_split(list(range(tensor.shape[0])), test_size=0.2, random_state=42)
    X_train = tensor[train_ind, :, :]
    y_train = classes[train_ind, :]
    X_test = tensor[test_ind, :, :]
    y_test = classes[test_ind, :]

    # Create VAE object, compile and fit the network
    vae = DenseVAE(tensor=X_train, **cfg["architecture"])
    vae.compile(optimizer="adam")
    vae.build((None, X_train.shape[1], X_train.shape[2]))
    history = vae.fit(X_train, validation_data=(X_test,), **cfg["fitting"])

    modelling.plot_history(history, MODEL_ID)


def _stack_all_conditions_data(conditions: list[str], df_list: list[pd.DataFrame], partition_size: int,
                               fixed_size: int):
    """
    Preprocesses each condition and stores the results in pp_data: dict. Then concatenates the data for all condition
    in one single tensor(s) or pd.DataFrames for inputting to the network.
    :param conditions: list[str] -- List of Conditions, e.g. ["MathxHard"]
    :param df_list: list[pd.DataFrame -- List of pd.Dataframes resulted from the dataloader.load_data()
    :param partition_size: int -- Size of window to take as the time_steps
    :param fixed_size: int -- Cutting data to this fixed_size to avoid padding / get equal length data
    :return: Union[dict, np.array, np.array, pd.DataFrame]
    """
    pp_data = {}

    for i, condition in enumerate(conditions):
        preprocessor = Preprocesser(df_list=df_list, condition=condition,
                                    partition_size=partition_size, fixed_size=fixed_size)
        tensor: np.array = preprocessor.preprocess_data()
        FEATURE_NAMES = list(preprocessor.df_list_processed[0].columns)

        pp_data[condition] = {
            "tensor": tensor,
            "y_list": preprocessor.y_list_processed,
            "condition": condition,
            "condition_encoded": i,
            "condition_df": pd.DataFrame([condition for n in range(0, tensor.shape[0])]),
            "condition_encoded_df": pd.DataFrame([i for n in range(0, tensor.shape[0])]),
            "feature_names": FEATURE_NAMES,
            "scaler_object": preprocessor.scaler
        }

    # Stack data of each condition to one single 3D numpy array / or pd.DataFrame
    tensor = np.concatenate([pp_data[key]["tensor"] for key in pp_data.keys()], axis=0)
    y_tensor = np.concatenate([pp_data[key]["y_list"] for key in pp_data.keys()], axis=0)
    conditions_df = pd.concat(pp_data[key]["condition_df"] for key in pp_data.keys())

    return pp_data, tensor, y_tensor, conditions_df


def _process_labels(y_tensor: np.array) -> Union[np.array, np.array]:
    """
    Helper function to discretize labels
    :param y_tensor: np.array
    :return: y_label, classes: Union[np.array, np.array] -- Labels and classes both in shape (N_SAMPLES, 1)
    """
    # Get Labels in Shape (1290 x 1), i.e. (Number of Samples, Dimension_Label)
    y_label_list = []
    for i in range(0, y_tensor.shape[0]):
        y = y_tensor[i][0]
        y_label_list.append(y)
    y_label = np.array(y_label_list)

    # Discretize labels into classes
    def _discretize_target(val: float) -> np.array:
        """ Call this function via np.vectorize(_discretize_target)(y_label) """
        if val < 0.1:
            return 0
        elif 0.1 <= val < 0.3:
            return 1
        else:
            return 2

    classes = np.vectorize(_discretize_target)(y_label)
    return y_label, classes


if __name__ == '__main__':
    main()
