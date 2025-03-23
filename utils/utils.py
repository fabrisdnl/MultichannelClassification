from torch.utils.data import DataLoader
import pickle
import cv2
import os
import h5py
import numpy as np


def load_mat_data(data_dir, label_txt=False):
    """
    Load images from a MATLAB .mat file and labels from either the .mat file or a dynamically determined .txt file.

    Args:
        data_dir (str): Path to the MATLAB data file.
        label_txt (bool): If True, load labels from a dynamically determined .txt file; otherwise, use numerical labels from .mat.

    Returns:
        tuple: (train_images, test_images, labels)
    """
    with h5py.File(data_dir, 'r') as f:
        # Load training and test images
        train_images = np.array(f['augTrainingImages'])  # Shape: (N, C, H, W)
        test_images = np.array(f['augTestImages'])  # Shape: (M, C, H, W)

        if label_txt:
            # Extract the last number in the filename (after 'fold_')
            filename = os.path.basename(data_dir)
            fold_number = ""
            if "fold_" in filename:
                parts = filename.split("fold_")
                if len(parts) > 1:
                    fold_number = "".join(filter(str.isdigit, parts[-1]))

            if fold_number:
                labels_path = os.path.join(os.path.dirname(data_dir), f"label_{fold_number}.txt")
                labels_path = os.path.normpath(labels_path)

                if os.path.exists(labels_path):
                    print(f"Loading labels from {labels_path}")
                    labels = np.loadtxt(labels_path, dtype=int)
                else:
                    raise FileNotFoundError(f"Labels file not found: {labels_path}")
            else:
                raise ValueError("Fold number not found in filename.")
        else:
            print("Loading labels from .mat file.")
            labels = np.array(f['label']).flatten()

    # Debugging prints
    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Labels shape: {labels.shape}")

    train_images = train_images.astype(np.float32, copy=False)
    test_images = test_images.astype(np.float32, copy=False)

    return train_images, test_images, labels


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
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, pin_memory=True),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, pin_memory=True),
    }


def compute_mean_std_mat(images):
    """
    Compute mean adn std for each band of dataset.

    Args:
        images (np.ndarray): Images array (N, C, H, W).

    Returns:
        tuple: (mean, std), each of length C (number of bands/channels).
    """
    mean = np.mean(images, axis=(0, 2, 3))
    std = np.std(images, axis=(0, 2, 3))

    return mean, std


def resize_images(images, new_size=(128, 128), interpolation=cv2.INTER_CUBIC):
    """
    Resize a batch of multispectral images while preserving details.

    Args:
        images (numpy.ndarray): Array of shape (N, C, H, W), where:
                                - N = number of images
                                - C = number of channels
                                - H, W = original image size
        new_size (tuple): Target size for resizing (height, width).
        interpolation (int): OpenCV interpolation method (default: Bicubic).

    Returns:
        numpy.ndarray: Resized batch of images with shape (N, C, new_H, new_W).
    """
    N, C, H, W = images.shape
    resized_images = np.zeros((N, C, new_size[0], new_size[1]), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            resized_images[i, c] = cv2.resize(images[i, c], new_size, interpolation=interpolation)

    return resized_images


def compute_mean_std(train_images, bands):
    """
    Compute for each spectral band the mean and the standard deviation from the training images.

    Args:
        train_images (list): List of train images, where each image is a 3D numpy array.
        bands (int): Number of spectral bands.

    Returns:
        dict: Dictionary with 'mean' and 'std' as numpy arrays of type float32.
    """
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


def save_metrics(metrics, filename):
    """
    Save metrics (such as loss and accuracy) to a file for later use.

    Args:
        metrics (dict): Dictionary containing the metrics to save,
                        typically {'train_loss': ..., 'train_accuracy': ...,
                                  'val_loss': ..., 'val_accuracy': ...}.
        filename (str): Path to the file where metrics will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(metrics, f)


def load_metrics(filename):
    """
    Load previously saved metrics from a file.

    Args:
        filename (str): Path to the file containing saved metrics.

    Returns:
        dict: Dictionary containing the loaded metrics.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

