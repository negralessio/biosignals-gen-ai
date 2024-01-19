""" Module that provides helper functions for modelling """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import logging
import src.utils as utils

utils.setup_logging()
logger = logging.getLogger(__name__)


def plot_history(history, model_id: str) -> None:
    """
    Plots the learning curves and saves the figures in assets
    :param history: history object -- Returned from vae.fit()
    :param model_id: str -- Unique Identifier of the current run / model
    :return: None
    """
    loss_dict = history.history

    loss_train = loss_dict["loss"]
    loss_val = loss_dict["val_loss"]

    rec_loss = loss_dict["reconstruction_loss"]
    rec_loss_val = loss_dict["val_reconstruction_loss"]

    kl_loss = loss_dict["kl_loss"]
    kl_loss_val = loss_dict["val_kl_loss"]

    x = [i + 1 for i in range(0, len(loss_train))]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].set_title("Total Loss", size=16, fontweight="bold")
    sns.lineplot(x=x, y=loss_train, label="Train", ax=axes[0])
    sns.lineplot(x=x, y=loss_val, label="Val", ax=axes[0])

    axes[1].set_title("Reconstruction Loss", size=16, fontweight="bold")
    sns.lineplot(x=x, y=rec_loss, label="Train", ax=axes[1])
    sns.lineplot(x=x, y=rec_loss_val, label="Val", ax=axes[1])

    axes[2].set_title("KL Loss", size=16, fontweight="bold")
    sns.lineplot(x=x, y=kl_loss, label="Train", ax=axes[2])
    sns.lineplot(x=x, y=kl_loss_val, label="Val", ax=axes[2])
    fig.tight_layout()

    saving_name: str = f"history-{model_id}.png"
    plt.savefig(f"assets/{saving_name}", bbox_inches='tight')
    logger.info(f"Saved plot in assets as '{saving_name}' ...")

