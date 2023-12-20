import yaml
import argparse
import logging
import datetime


def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file as a dict.

    :param config_path: str -- Path to the configuration file
    :return: dict -- Configuration dictionary
    """
    try:
        with open(config_path, "r") as yamlfile:
            return yaml.load(yamlfile, yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {config_path} not found!")
    except PermissionError:
        raise PermissionError(f"Insufficient permission to read {config_path}!")
    except IsADirectoryError:
        raise IsADirectoryError(f"{config_path} is a directory!")


def parse_config_file() -> tuple[dict, str]:
    """
    Function that parses the config.yaml

    :return:    cfg: dict -- Parsed config file as dict
                args.config: str -- Config path
    """
    # Parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)
    return cfg, args.config


def setup_logging(loglevel=logging.INFO) -> None:
    """ Handles the logger setup / configuration

    :param loglevel: Level of logging, e.g. {logging.DEBUG, logging.INFO}
    :return: None
    """
    logging.basicConfig(level=loglevel, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_time_date() -> str:
    """ Returns the datetime as str

    :return: time_str: str
    """
    time_str = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    return time_str


def get_model_id(n_epochs, n_batches, val_split, latent_dim) -> str:
    """ Returns an ID for a model run using the current time + hyperparams

    :param n_epochs: int -- # of epochs
    :param n_batches: int -- # of batches
    :param val_split: float -- validation size (e.g. 0.2)
    :param latent_dim: int -- # of latent dimension
    :return: result: str -- Model ID given the input params
    """
    time = get_time_date()
    result = f"T{time}---E{n_epochs}-B{n_batches}-VAL{int(val_split*100)}-LD{latent_dim}"
    return result
