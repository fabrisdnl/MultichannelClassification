import torch
from torchvision import transforms
import numpy as np


# # Custom transform class for handling 13-channel EuroSAT images
# class EuroSATTransform:
#     def __init__(self, normalize=True, resize=(64, 64), augment=True):
#         self.normalize = normalize
#         self.resize = resize
#         self.augment = augment
#
#         # Define actual means and stds for each of the 13 channels
#         self.means = torch.tensor([1004.0, 1183.0, 1043.0, 952.0, 1244.0, 1678.0, 2193.0, 2260.0, 2088.0, 1595.0, 1313.0, 1165.0, 870.0])
#         self.stds = torch.tensor([507.0, 457.0, 443.0, 457.0, 515.0, 733.0, 1098.0, 1199.0, 1098.0, 923.0, 782.0, 690.0, 463.0])
#
#     def __call__(self, img):
#         # Convert to tensor of shape: 13 x 64 x 64
#         img = torch.tensor(np.array(img)).float()
#
#         # Optional data augmentations
#         if self.augment:
#             img = self.augment_data(img)
#
#         # Resize (if specified)
#         if self.resize:
#             img = self.resize_image(img, self.resize)
#
#         # Normalize each channel independently
#         if self.normalize:
#             img = self.normalize_bands(img)
#
#         return img
#
#     def augment_data(self, img):
#         # Apply random augmentations like flips and rotations
#         if np.random.rand() > 0.5:
#             img = img.flip(-1)  # Horizontal flip
#         if np.random.rand() > 0.5:
#             img = img.flip(-2)  # Vertical flip
#         if np.random.rand() > 0.5:
#             img = torch.rot90(img, k=1, dims=(-2, -1))  # 90-degree rotation
#
#         return img
#
#     def normalize_bands(self, img):
#         # Normalize image using mean and std for each band
#         img = (img - self.means[:, None, None]) / self.stds[:, None, None]
#         return img
#
#     def resize_image(self, img, size):
#         # Resize each channel separately to the target size
#         img_resized = torch.stack([transforms.functional.resize(img[i], size) for i in range(img.shape[0])])
#         return img_resized

# Custom Transform class that normalizes the 13-channel image
class EuroSATTransform:
    def __init__(self, mean, std, augment=False):
        self.mean = mean
        self.std = std
        self.augment = augment

    def __call__(self, image):
        # Normalize each channel
        for i in range(len(self.mean)):
            image[i] = (image[i] - self.mean[i]) / self.std[i]

        # Augmentations if enabled
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [1])  # Horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [2])  # Vertical flip

        return image