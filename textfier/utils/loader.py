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

    # Tries to load the file
    try:
        # Loads a .csv file using Pandas
        csv = pd.read_csv(file_name)

        # Gathers the labels
        labels = csv['label'].tolist()

        # Gathers the samples
        samples = csv['sample'].tolist()

        logger.debug('Labels and samples loaded.')

        return labels, samples

    # If file can not be loaded
    except FileNotFoundError:
        # Creates an error
        e = f'File not found: {file_name}.'

        # Logs the error
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

    # Tries to load the file
    try:
        # Opens the .txt file
        file = open(file_name, 'rb')

        # Reads the text
        text = file.read().decode(encoding='utf-8')

        logger.debug('Text loaded.')

        return text

    # If file can not be loaded
    except FileNotFoundError:
        # Creates an error
        e = f'File not found: {file_name}.'

        # Logs the error
        logger.error(e)

        raise
