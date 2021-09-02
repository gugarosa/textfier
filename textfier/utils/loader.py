"""Data-loading utilities.
"""

import pandas as pd

import textfier.utils.logging as l

logger = l.get_logger(__name__)


def load_csv(file_name):
    """Loads a .csv file using Pandas.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        Lists of samples and their labels.

    """

    logger.debug('Loading labels and samples from: %s ...', file_name)

    try:
        csv = pd.read_csv(file_name)

        labels = csv['label'].tolist()
        samples = csv['sample'].tolist()

        logger.debug('Labels and samples loaded.')

        return samples, labels

    except FileNotFoundError:
        e = f'File not found: {file_name}.'

        logger.error(e)

        raise


def load_txt(file_name):
    """Loads a .txt file.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        A string with the loaded text.

    """

    logger.debug('Loading text from: %s ...', file_name)

    try:
        file = open(file_name, 'rb')

        text = file.read().decode(encoding='utf-8')

        logger.debug('Text loaded.')

        return text

    except FileNotFoundError:
        e = f'File not found: {file_name}.'

        logger.error(e)

        raise
