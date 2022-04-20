"""Data-loading utilities.
"""

from typing import List, Tuple

import pandas as pd

from textfier.utils import logging

logger = logging.get_logger(__name__)


def load_csv(file_name: str) -> Tuple[List[str], List[int]]:
    """Loads a .csv file using Pandas.

    Args:
        file_name: The file name to be loaded.

    Returns:
        (Tuple[List[str], List[int]]): Samples and their labels.

    """

    logger.debug("Loading labels and samples from: %s ...", file_name)

    try:
        csv = pd.read_csv(file_name)

        labels = csv["label"].tolist()
        samples = csv["sample"].tolist()

        logger.debug("Labels and samples loaded.")

        return samples, labels

    except FileNotFoundError:
        e = f"File not found: {file_name}."

        logger.error(e)

        raise


def load_txt(file_name: str) -> str:
    """Loads a .txt file.

    Args:
        file_name: The file name to be loaded.

    Returns:
        (str): Loaded text.

    """

    logger.debug("Loading text from: %s ...", file_name)

    try:
        file = open(file_name, "rb")

        text = file.read().decode(encoding="utf-8")

        logger.debug("Text loaded.")

        return text

    except FileNotFoundError:
        e = f"File not found: {file_name}."

        logger.error(e)

        raise
