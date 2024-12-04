import torch
from torch.utils.data import Dataset


class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT data with 13-band images.

    Accepts preloaded data and applies transformations.

    Args:
        data (numpy array): Preloaded images.
        labels (list[int]): Numeric labels corresponding to each image.
        transform (callable, optional): Transformations to apply to the data.
    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the 13-band image
        image = self.data[idx]

        # Access the numeric label
        label = self.labels[idx]

        # Ensure the image has 13 channels
        if image.shape[0] != 13:
            raise ValueError(f"Expected image shape [13, height, width], but got {image.shape}")

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        # Convert the image and label to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label).long()

        return image, label
