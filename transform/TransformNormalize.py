import numpy as np
import torch


class TransformNormalize:
    """
    Normalization for multispectral images.

    Args:
        mean (np.ndarray): Array of mean values for each band.
        std (np.ndarray): Array of standard deviation values for each band.
    """

    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __call__(self, image):
        """
        Apply normalization to an image.

        Args:
            image (np.ndarray): Array of shape (C, H, W).

        Returns:
            torch.tensor: Normalized image tensor.
        """

        image = image.astype(np.float32)

        normalized_image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return torch.tensor(normalized_image, dtype=torch.float32)
