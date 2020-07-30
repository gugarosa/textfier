from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """
    """

    def __init__(self, data, mask, labels):
        """
        """

        #
        self.data = data

        #
        self.mask = mask

        #
        self.labels = labels

    def __getitem__(self, idx):
        """
        """

        return {
            'input_ids': self.data[idx],
            'attention_mask': self.mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        """
        """

        return len(self.data)