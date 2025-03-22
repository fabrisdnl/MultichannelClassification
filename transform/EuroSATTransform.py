import numpy as np
import torch
import torch.nn.functional as F


class EuroSATTransform:
    """
    Data augmentation and normalization for multispectral images.

    Args:
        mean (np.ndarray): Array of mean values for each band.
        std (np.ndarray): Array of standard deviation values for each band.
        augment (bool): Whether to apply data augmentation.
    """

    def __init__(self, mean, std, augment=False):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment

    def __call__(self, image):
        """
        Apply normalization and optional data augmentation.

        Args:
            image (np.ndarray): Multispectral image array with shape (C, height, width).

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        # Convert image to float32 for normalization
        image = image.astype(np.float32)

        # Normalize each channel
        for i in range(13):
            image[i] = (image[i] - self.mean[i]) / self.std[i]

        # Ensure the image is contiguous in memory
        image = np.ascontiguousarray(image)

        # Convert to tensor
        image_tensor = torch.tensor(image).float()

        # Apply data augmentation if enabled
        if self.augment:
            _, height, width = image_tensor.shape

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])

            # Random rotation (90, 180, 270 degrees)
            if torch.rand(1).item() > 0.5:
                k = torch.randint(1, 4, (1,)).item()
                image_tensor = torch.rot90(image_tensor, k=k, dims=(1, 2))

            # Random crop and resize back to original size
            if torch.rand(1).item() > 0.5:
                crop_size = int(0.9 * min(height, width))
                top = torch.randint(0, height - crop_size, (1,)).item()
                left = torch.randint(0, width - crop_size, (1,)).item()
                image_tensor = image_tensor[:, top:top + crop_size, left:left + crop_size]

                # Resize back to original size
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False
                ).squeeze(0)

            # Add random Gaussian noise
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(image_tensor) * 0.02
                image_tensor += noise

        return image_tensor
