import torch
from torch.utils.data import Dataset


class MatDataset(Dataset):
    """
    PyTorch Dataset for Mat file data with multi-band images.

    Accepts preloaded data and applies transformations.

    Args:
        data (numpy array): Normalized images (N, C, H, W).
        labels (numpy array, optional): Numeric labels corresponding to each image.
        transform (callable, optional): Optional transformations.
    """

    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        image = image.clone().detach().to(torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image
