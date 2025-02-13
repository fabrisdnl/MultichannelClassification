import numpy as np
import torch


class EuroSATNormalize:
    """
    Normalization for multispectral images with 13 channels.

    Args:
        mean (np.ndarray): Array of mean values for each band (length 13).
        std (np.ndarray): Array of standard deviation values for each band (length 13).
    """

    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __call__(self, image):
        """
        Apply normalization to an image.

        Args:
            image (np.ndarray): Array of shape (13, H, W).

        Returns:
            torch.tensor: Normalized image tensor.
        """
        if image.shape[0] != 13:
            raise ValueError(f"Expected 13 bands, but got {image.shape[0]} bands")

        image = image.astype(np.float32)

        normalized_image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return torch.tensor(normalized_image, dtype=torch.float32)
