"""Default dataset class.
"""

import torch
from torch.utils.data import Dataset

import textfier.utils.logging as l

logger = l.get_logger(__name__)


class TextClassificationDataset(Dataset):
    """TextClassificationDataset implements a dataset used to handle customizable text classification tasks.

    """

    def __init__(self, data, mask, labels):
        """Initialization method.

       Args:
            data (list): Encoded data already pre-processed by a tokenizer.
            mask (list): Attention masks already pre-processed by a tokenizer.
            labels (list): Tasks labels.

        """

        logger.debug('Creating dataset ...')

        # Encoded data
        self.data = data

        # Attention masks
        self.mask = mask

        # Tasks labels
        self.labels = torch.tensor(labels)

        logger.debug('Dataset created.')

    def __getitem__(self, idx):
        """Private method that is the base for PyTorch's iterator.

        Args:
            idx (int): Index of sample.

        Returns:
            A dictionary containing `input_ids`, `attention_mask` and `labels`.

        """

        return {
            'input_ids': self.data[idx],
            'attention_mask': self.mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        """Private method that is the base for PyTorch's iterator.

        """

        return len(self.data)
