import os
import h5py
import ast
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader
import pickle
from sklearn.decomposition import PCA
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
                labels_path = os.path.normpath(labels_path)  # Ensure correct path format

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

    # Convert images to float32 for deep learning compatibility
    train_images = train_images.astype(np.float32, copy=False)
    test_images = test_images.astype(np.float32, copy=False)

    return train_images, test_images, labels


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


def apply_pca(train_data, test_data, num_components=4):
    N, C, H, W = train_data.shape
    M, _, _, _ = test_data.shape

    # Reshape to apply PCA on channels
    train_reshaped = train_data.reshape(N, C, -1).transpose(0, 2, 1)  # (N, H*W, C)
    test_reshaped = test_data.reshape(M, C, -1).transpose(0, 2, 1)

    pca = PCA(n_components=num_components)

    train_pca = np.array([pca.fit_transform(img) for img in train_reshaped])  # (N, H*W, num_components)
    test_pca = np.array([pca.transform(img) for img in test_reshaped])

    # Reshape to original shape
    train_pca = train_pca.transpose(0, 2, 1).reshape(N, num_components, H, W)
    test_pca = test_pca.transpose(0, 2, 1).reshape(M, num_components, H, W)

    return train_pca, test_pca


def create_compressed_dataset(dataset_path, compressed_path):
    """
    Creates a compressed HDF5 dataset from the raw image dataset.

    Args:
        dataset_path (str): Path to the raw dataset.
        compressed_path (str): Path to save the compressed dataset.
    """
    print("Compressing dataset into HDF5 format...")
    compress_to_hdf5(dataset_path, compressed_path)
    print(f"Compressed dataset saved at {compressed_path}.")


def load_compressed_dataset(compressed_path):
    """
    Loads images and labels from the compressed HDF5 dataset.

    Args:
        compressed_path (str): Path to the compressed dataset.

    Returns:
        tuple: Loaded images, labels, and class map.
    """
    print("Loading compressed dataset from HDF5 format...")
    return load_from_hdf5(compressed_path)


def compress_to_hdf5(dataset_path, compressed_path):
    """
    Compress a dataset of multispectral TIFF images into an HDF5 file.

    Args:
        dataset_path (str): Path to the dataset directory with TIFF images organized by class.
        compressed_path (str): Path to the output compressed HDF5 file.
    """
    images = []
    labels = []
    class_map = {}
    current_label = 0

    # Scan directories to collect TIFF images and assign labels.
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Map class name to a label.
        class_map[class_name] = current_label

        for img_file in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_file)
            if img_file.endswith('.tif'):
                try:
                    img = tiff.imread(img_path)
                    images.append(img)
                    labels.append(current_label)
                except Exception as e:
                    print(f"Error loading image: {img_path}. {e}")

        current_label += 1

    images = np.array(images, dtype=np.uint16)
    labels = np.array(labels, dtype=np.int64)

    # Write the data to an HDF5 file with gzip compression.
    with h5py.File(compressed_path, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip', compression_opts=9)
        f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        f.create_dataset('class_map', data=np.string_(str(class_map)))

    print(f"Compressed dataset saved to {compressed_path}.")


def load_from_hdf5(compressed_path):
    """
    Load multispectral images and labels from a compressed HDF5 file.

    Args:
        compressed_path (str): Path to the compressed HDF5 file.

    Returns:
        tuple: A tuple containing:
            - images (np.ndarray): Multispectral images (e.g., 13-band).
            - labels (np.ndarray): Class labels corresponding to images.
            - class_map (dict): Mapping of class names to labels.
    """
    with h5py.File(compressed_path, 'r') as f:
        # Load images
        images = np.array(f['images'])
        # print(images.shape)
        if images.ndim == 4 and images.shape[-1] == 13:
            images = images.transpose(0, 3, 1, 2)

        # Load labels
        labels = np.array(f['labels'])

        # Load and decode the class map
        class_map_str = f['class_map'][()].decode('utf-8')
        class_map = ast.literal_eval(class_map_str)

    return images, labels, class_map


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

    for i in range(N):  # Iterate over images
        for c in range(C):  # Iterate over channels
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

