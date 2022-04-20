"""Default dataset class.
"""

from typing import Any, Dict

import torch

from textfier.utils import logging

logger = logging.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Dataset implements a default class used to handle customizable datasets."""

    def __init__(self, **kwargs):
        """Initialization method."""

        logger.debug("Creating dataset ...")

        for (key, value) in kwargs.items():
            setattr(self, key, value)

        logger.debug("Dataset created.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Private method that serves as PyTorch's iterator.

        Args:
            idx: Index of sample.

        Returns:
            (Dict[str, Any]): Desired keys/values pair.

        """

        sample = {}

        for (key, value) in vars(self).items():
            sample[key] = value[idx]

        return sample

    def __len__(self) -> int:
        """Private method that serve as PyTorch's auxiliary.

        Returns:
            (int): Length of the first dataset's property.

        """

        prop = list(vars(self).values())[0]

        return len(prop)
