import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.model_selection import train_test_split
from base.EuroSATDataset import EuroSATDataset
from typing import List, Tuple, Dict


def load_all_images(data_dir):
    """
    Load multispectral images as NumPy arrays and their corresponding integer labels from the dataset directory.
    Assign a unique integer index to each class label.

    Args:
        data_dir (str): Path to the EuroSAT dataset directory.

    Returns:
        list, list, dict: List of image arrays, their corresponding integer labels,
                          and a mapping of class names to integer indices.
    """
    images = []
    labels = []

    # Assign integer indices to class names
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            if os.path.isfile(img_path):
                # Load the image as a NumPy array
                try:
                    image_array = load_image_numpy(img_path)
                    images.append(image_array)
                    labels.append(class_map[class_name])
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return images, labels, class_map


def load_image_torch(image_path):
    """
    Load a multispectral image as a PyTorch tensor with shape (13, height, width).

    Args:
        image_path: Path to the multispectral image.

    Returns:
        torch.Tensor: Multispectral image tensor with shape (13, height, width).
    """
    # Read the TIFF file
    image = tiff.imread(image_path)
    # If the last dimension is the number of bands, transpose it
    if image.shape[-1] == 13:
        image = image.transpose(2, 0, 1)

    # Convert to PyTorch tensor
    image = torch.tensor(image, dtype=torch.float32)
    
    return image


def load_image_numpy(image_path):
    """
    Load a multispectral image as a NumPy array with shape (13, height, width).

    Args:
        image_path (str): Path to the multispectral image.

    Returns:
        np.ndarray: Multispectral image array with shape (13, height, width).
    """
    # Read the TIFF file
    image = tiff.imread(image_path)
    # If the last dimension is the number of bands, transpose it
    if image.shape[-1] == 13:
        image = image.transpose(2, 0, 1)

    # Convert to NumPy array of type float32
    image = np.array(image, dtype=np.float32)

    return image


def split_data_preloaded(images, labels, train_split, validation_split):
    """
    Splits the preloaded data into train, validation, and test sets.

    Args:
        images (list): List of all loaded images.
        labels (list): List of corresponding labels.
        train_split (float): Fraction of data to allocate for training.
        validation_split (float): Fraction of training data for validation.

    Returns:
        dict: Contains train, validation, and test splits.
    """
    # Generate casual indices
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    # Fix dimensions
    train_size = int(len(images) * train_split)
    validation_size = int(train_size * validation_split)

    # Initialize splits
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]

    return {
        "train": {
            "data": [images[i] for i in train_indices],
            "labels": [labels[i] for i in train_indices],
        },
        "validation": {
            "data": [images[i] for i in validation_indices],
            "labels": [labels[i] for i in validation_indices],
        },
        "test": {
            "data": [images[i] for i in test_indices],
            "labels": [labels[i] for i in test_indices],
        },
    }


def create_datasets(splits, train_transform, val_test_transform, val_split=0.1):
    """
    Creates train, validation, and test datasets from a given split.

    Args:
        splits (dict): Dictionary containing 'train' and 'test' (image paths and labels).
        train_transform (callable): Transformations for the training set.
        val_test_transform (callable): Transformations for validation and test sets.
        val_split (float): Fraction of the training set to use as validation set.

    Returns:
        dict: Dictionary with 'train', 'validation', and 'test' datasets.
    """
    train_image_paths, train_labels = splits["train"]
    test_image_paths, test_labels = splits["test"]

    # Split the training set into training and validation sets
    total_train = len(train_image_paths)
    val_size = int(val_split * total_train)
    train_size = total_train - val_size

    val_image_paths = train_image_paths[train_size:]
    val_labels = train_labels[train_size:]
    train_image_paths = train_image_paths[:train_size]
    train_labels = train_labels[:train_size]

    # Create datasets
    return {
        "train": EuroSATDataset(train_image_paths, train_labels, transform=train_transform),
        "validation": EuroSATDataset(val_image_paths, val_labels, transform=val_test_transform),
        "test": EuroSATDataset(test_image_paths, test_labels, transform=val_test_transform),
    }


def create_dataloaders(datasets, batch_size):
    """
    Creates dataloaders from the provided datasets.

    Args:
        datasets (dict): Dictionary containing 'train', 'validation', and 'test' datasets.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        dict: Dictionary with 'train', 'validation', and 'test' dataloaders.
    """
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def compute_mean_std(train_images, bands):
    """
    Compute for each spectral band the mean and the standard deviation from the training images.

    Args:
        train_images (list): List of train images, where each image is a 3D numpy array.
        bands (int): Number of spectral bands.

    Returns:
        dict: Dictionary with 'mean' and 'std' as numpy arrays of type float32.
    """
    # Print length for debugging
    print(f"Number of training images: {len(train_images)}")

    # Initialize arrays for mean and standard deviation
    mean = np.zeros(bands, dtype=np.float32)
    std = np.zeros(bands, dtype=np.float32)

    # Compute mean and std for each band
    for i in range(bands):
        band_values = np.concatenate([img[:, :, i].flatten() for img in train_images])
        mean[i] = np.mean(band_values)
        std[i] = np.std(band_values)

    return {
        "mean": mean,
        "std": std
    }
