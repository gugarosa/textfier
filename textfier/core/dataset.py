"""Default dataset class.
"""

import torch

import textfier.utils.logging as l

logger = l.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Dataset implements a default class used to handle customizable datasets.

    """

    def __init__(self, **kwargs):
        """Initialization method.

        """

        logger.debug('Creating dataset ...')

        # For every key-value pair
        for (key, value) in kwargs.items():
            # Sets an attribute based on key-value pair
            setattr(self, key, value)

        # Key-value pair can be re-utilized to derive
        # the length of the dataset
        self.length = len(getattr(self, key))

        logger.debug('Dataset created.')

    def __getitem__(self, idx):
        """Private method that serves as PyTorch's iterator.

        Args:
            idx (int): Index of sample.

        Returns:
            A dictionary containing desired keys.

        """

        # Defines an empty dictionary for holding the sample
        sample = {}

        # For every key-value pair
        for (key, value) in vars(self).items():
            # Adds the key-value pair based on current index
            sample[key] = value[idx]

        return sample

    def __len__(self):
        """Private method that serve as PyTorch's auxiliary.

        """

        return self.length
