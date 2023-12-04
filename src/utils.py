import yaml
import argparse
import logging


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
