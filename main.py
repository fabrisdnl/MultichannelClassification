import argparse
import sys
import os
from utils import utils, metrics
from base.EuroSATDataset import EuroSATDataset
from base.EuroSATMatDataset import EuroSATMatDataset
from transform.EuroSATTransform import EuroSATTransform
from transform.EuroSATNormalize import EuroSATNormalize
from model.HybridModel import HybridModel
from model.AdaptedHybridModel import AdaptedHybridModel
from trainer import train, test, test_logits, train_only
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np


def create_arg_parser():
    """
    Creates and returns the ArgumentParser object for command-line arguments.

    Returns:
        argparse.ArgumentParser: Argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description='Land Cover Classification using Sentinel-2 data.')
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset directory containing images.')
    parser.add_argument('--saves_dir', default="saves", help='Path to save models and logs (default: "saves").')
    parser.add_argument('--load_saved_models', action='store_true', help='Load pre-trained models if available.')
    parser.add_argument('--compressed_dataset', action='store_true', help='Indicates if the dataset is in compressed '
                                                                          'HDF5 format.')
    parser.add_argument('--mat_format', action='store_true', help='Indicates if the dataset is in mat format.')
    parser.add_argument('--labels_txt', action='store_true', help='Indicates if the labels must be read on txt.')
    parser.add_argument('--no_valid', action='store_true', help='Indicates if ignore validation and do only training.')
    parser.add_argument('--directory', action='store_true', help='Indicates if input is a folder o mat files.')

    return parser


def process_mat_no_valid(dataloaders, device, num_channels, save_dir, load_saved_models,
                         unique_labels, file_name_without_ext):
    """
    Process: trains the model or loads a pre-trained model.

    Args:
        dataloaders (dict): Dataloaders for train, validation, and test.
        device (torch.device): Device to use for computations.
        num_channels (int): Number of bands of images.
        save_dir (str): Directory where models will be saved.
        load_saved_models (bool): Whether to load pre-trained models.
        unique_labels (list): List of unique labels.
        file_name_without_ext (str): Name of the input file.

    Returns:
        dict: Dictionary containing metrics and logits.
    """
    model_path = os.path.join(save_dir, f"model_{file_name_without_ext}.pth")
    metrics_path = os.path.join(save_dir, f"metrics_{file_name_without_ext}.pkl")
    model = HybridModel(num_channels=num_channels, num_classes=len(unique_labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    # Train or load the model
    if load_saved_models and os.path.exists(model_path):
        print(f"Loading saved model for {file_name_without_ext} ...")
        model.load_state_dict(torch.load(model_path))
        # Load training metrics if model is loaded
        metrics_loaded = utils.load_metrics(metrics_path)
        train_losses = metrics_loaded['train_losses']
        train_accuracies = metrics_loaded['train_accuracies']
    else:
        print(f"Training model for {file_name_without_ext} ...")
        metrics_computed = train_only.train_model(
            model, dataloaders["train"], criterion, optimizer, scheduler, device,
            num_epochs=25
        )
        train_losses = metrics_computed['train_losses']
        train_accuracies = metrics_computed['train_accuracies']
        utils.save_metrics(metrics_computed, metrics_path)
        print(f"Metrics of training for {file_name_without_ext} saved at {metrics_path}")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {file_name_without_ext} saved at {model_path}")

    # Test the model
    print(f"Testing model...")
    logits = test_logits.test_model(model, dataloaders["test"], device, save_dir, file_name_without_ext)

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "logits": logits
    }


def process_mat(dataloaders, device, num_channels, save_dir, load_saved_models, unique_labels, file_name_without_ext):
    """
    Process: trains and evaluates the model or loads a pre-trained model.

    Args:
        dataloaders (dict): Dataloaders for train, validation, and test.
        device (torch.device): Device to use for computations.
        num_channels (int): Number of bands of images.
        save_dir (str): Directory where models will be saved.
        load_saved_models (bool): Whether to load pre-trained models.
        unique_labels (list): List of unique labels.
        file_name_without_ext (str): Name of the input file.

    Returns:
        dict: Dictionary containing metrics and logits.
    """
    model_path = os.path.join(save_dir, f"model_{file_name_without_ext}.pth")
    metrics_path = os.path.join(save_dir, f"metrics_{file_name_without_ext}.pkl")
    model = AdaptedHybridModel(num_channels=num_channels, num_classes=len(unique_labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    # Train or load the model
    if load_saved_models and os.path.exists(model_path):
        print(f"Loading saved model for {file_name_without_ext} ...")
        model.load_state_dict(torch.load(model_path))
        # Load training metrics if model is loaded
        metrics_loaded = utils.load_metrics(metrics_path)
        train_losses = metrics_loaded['train_losses']
        train_accuracies = metrics_loaded['train_accuracies']
        val_losses = metrics_loaded['val_losses']
        val_accuracies = metrics_loaded['val_accuracies']
    else:
        print(f"Training model for {file_name_without_ext} ...")
        metrics_computed = train.train_model(
            model, dataloaders["train"], dataloaders["validation"], criterion, optimizer, scheduler, device,
            num_epochs=25
        )
        train_losses = metrics_computed['train_losses']
        train_accuracies = metrics_computed['train_accuracies']
        val_losses = metrics_computed['val_losses']
        val_accuracies = metrics_computed['val_accuracies']
        utils.save_metrics(metrics_computed, metrics_path)
        print(f"Metrics of training and validation for {file_name_without_ext} saved at {metrics_path}")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {file_name_without_ext} saved at {model_path}")

    # Test the model
    print(f"Testing model...")
    logits = test_logits.test_model(model, dataloaders["test"], device, save_dir, file_name_without_ext)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "logits": logits
    }


def process_split(split_name, dataloaders, device, output_dir, save_dir, load_saved_models, class_map):
    """
    Processes a single data split: trains and evaluates the model or loads a pre-trained model.

    Args:
        split_name (str): Name of the split ('80-20' or '50-50').
        dataloaders (dict): Dataloaders for train, validation, and test.
        device (torch.device): Device to use for computations.
        save_dir (str): Directory where models will be saved.
        load_saved_models (bool): Whether to load pre-trained models.
        class_map (dict): Mapping of class names to integer indices.

    Returns:
        dict: Dictionary containing metrics (test accuracy).
    """
    model_path = os.path.join(save_dir, f"model_split_{split_name}.pth")
    metrics_path = os.path.join(save_dir, f"metrics_split_{split_name}.pkl")
    model = AdaptedHybridModel(num_classes=len(class_map)).to(device)  # Use the number of classes from class_map
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    # Train or load the model
    if load_saved_models and os.path.exists(model_path):
        print(f"Loading saved model for {split_name} split...")
        model.load_state_dict(torch.load(model_path))
        # Load training metrics if model is loaded
        metrics_loaded = utils.load_metrics(metrics_path)
        train_losses = metrics_loaded['train_losses']
        train_accuracies = metrics_loaded['train_accuracies']
        val_losses = metrics_loaded['val_losses']
        val_accuracies = metrics_loaded['val_accuracies']
    else:
        print(f"Training model for {split_name} split...")
        metrics_computed = train.train_model(
            model, dataloaders["train"], dataloaders["validation"], criterion, optimizer, scheduler, device,
            num_epochs=25
        )
        train_losses = metrics_computed['train_losses']
        train_accuracies = metrics_computed['train_accuracies']
        val_losses = metrics_computed['val_losses']
        val_accuracies = metrics_computed['val_accuracies']
        utils.save_metrics(metrics_computed, metrics_path)
        torch.save(model.state_dict(), model_path)
        print(f"Model for {split_name} split saved at {model_path}.")

    # Test the model
    print(f"Testing model for {split_name} split...")
    test_accuracy, y_pred, y_true = test.test_model(model, dataloaders["test"], device)

    # Plot confusion matrix
    confusion_matrix_output_dir = os.path.join(output_dir, f"confusion_matrix_{split_name}")
    metrics.plot_confusion_matrix(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        classes=list(class_map.keys()),  # Use class names from class_map
        output_dir=confusion_matrix_output_dir
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
    }


def execute(data_dir, save_dir, load_saved_models, compressed_dataset, mat_format, labels_txt, no_valid):
    if mat_format:
        file_name = os.path.basename(data_dir)
        file_name_without_ext = os.path.splitext(file_name)[0]

        print(f"Loading MATLAB data from {data_dir}")
        train_data, test_data, labels = utils.load_mat_data(data_dir, labels_txt)

        num_channels = train_data.shape[1]
        print(f"Number of channels: {num_channels}")

        # Resize images if channels exceed 13
        # if num_channels > 13:
        #     train_data = utils.resize_images(train_data, new_size=(160, 160))
        #     test_data = utils.resize_images(test_data, new_size=(160, 160))

        # Compute mean and std on training images
        mean, std = utils.compute_mean_std_mat(train_data)

        # Normalize labels: MATLAB labels start from 1, convert to zero-based index
        labels -= 1

        # Get dataset sizes
        num_train = len(train_data)
        num_test = len(test_data)
        num_labels = len(labels)
        unique_labels = np.unique(labels)

        print(f"Unique labels: {unique_labels}")
        print(f"Number of train+validation images loaded: {num_train}")
        print(f"Number of test images loaded: {num_test}")
        print(f"Number of labels loaded: {num_labels}")

        if no_valid:
            # Resize images if channels exceed 13
            # if num_channels > 13:
            #     train_data = utils.resize_images(train_data, new_size=(160, 160))
            #     test_data = utils.resize_images(test_data, new_size=(160, 160))

            # No validation required
            train_images = train_data
            test_images = test_data

            normalize_transform = EuroSATNormalize(mean, std)
            print("Creating datasets and dataloaders...")
            datasets = {
                "train": EuroSATMatDataset(train_images, labels, transform=normalize_transform),
                "test": EuroSATMatDataset(test_images, transform=normalize_transform),
            }

            batch_size = 8
            dataloaders = {
                "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, pin_memory=True),
                "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, pin_memory=True),
            }

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            results = process_mat_no_valid(dataloaders, device, num_channels, save_dir,
                                  load_saved_models, unique_labels, file_name_without_ext)

            metrics.plot_metrics_train_only(
                train_losses=results["train_losses"],
                train_accuracies=results["train_accuracies"],
                output_dir=output_dir,
            )

            print("Execution terminated.")
        else:
            # Generate shuffled indices
            indices = np.arange(num_train)
            np.random.shuffle(indices)  # Shuffle the dataset

            # Compute split index for 90% training, 10% validation
            split_idx = int(num_train * 0.9)
            train_indices = indices[:split_idx]  # First 90% for training
            valid_indices = indices[split_idx:]  # Remaining 10% for validation

            # Apply shuffled indices to both images and labels
            train_images = train_data[train_indices]
            valid_images = train_data[valid_indices]
            test_images = test_data  # Test set remains unchanged

            train_labels = labels[train_indices]
            valid_labels = labels[valid_indices]

            # Print dataset shapes for verification
            print(f"Train set: {train_images.shape}, Labels: {train_labels.shape}")
            print(f"Validation set: {valid_images.shape}, Labels: {valid_labels.shape}")
            print(f"Test set: {test_images.shape}")

            # Extract filename without extension from `data_dir`
            base_filename = os.path.splitext(os.path.basename(data_dir))[0]

            # Define the filename with the dataset name
            indices_filename = os.path.join(os.path.dirname(data_dir), f"train_indices_{base_filename}.txt")

            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(indices_filename), exist_ok=True)

            # Save train_indices as a text file
            np.savetxt(indices_filename, train_indices, fmt="%d")

            print(f"Train indices saved to {indices_filename}")

            normalize_transform = EuroSATNormalize(mean, std)
            print("Creating datasets and dataloaders...")
            datasets = {
                "train": EuroSATMatDataset(train_images, train_labels, transform=normalize_transform),
                "validation": EuroSATMatDataset(valid_images, valid_labels, transform=normalize_transform),
                "test": EuroSATMatDataset(test_images, transform=normalize_transform),
            }
            if labels_txt:
                dataloaders = utils.create_dataloaders(datasets, batch_size=8)
            else:
                dataloaders = utils.create_dataloaders(datasets, batch_size=32)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            results = process_mat(dataloaders, device, num_channels, save_dir,
                                  load_saved_models, unique_labels, file_name_without_ext)

            metrics.plot_metrics_train(
                train_losses=results["train_losses"],
                val_losses=results["val_losses"],
                train_accuracies=results["train_accuracies"],
                val_accuracies=results["val_accuracies"],
                output_dir=output_dir,
            )

        print("Execution terminated.")

    else:
        if compressed_dataset:
            compressed_path = data_dir
            if not os.path.exists(compressed_path):
                raise FileNotFoundError(f"Compressed dataset not found at {compressed_path}.")
            print("Loading compressed images into memory...")
            all_images, all_labels, class_map = utils.load_compressed_dataset(compressed_path)
        else:
            print("Saving dataset in compressed HDF5 format...")
            compressed_path = os.path.join(os.path.dirname(data_dir), "dataset_compressed.h5")
            utils.create_compressed_dataset(data_dir, compressed_path)
            all_images, all_labels, class_map = utils.load_compressed_dataset(compressed_path)

        print(f"Number of images loaded: {len(all_images)}")
        print(f"Class map: {class_map}")

        # Create splits
        splits = {
            "80-20": utils.split_data_preloaded(all_images, all_labels, train_split=0.8, validation_split=0.1),
            "50-50": utils.split_data_preloaded(all_images, all_labels, train_split=0.5, validation_split=0.1),
        }

        # Compute normalization statistics and create transformations
        print("Calculating normalization statistics...")
        bands = 13
        transformations = {}
        for split_name, split_data in splits.items():
            stats = utils.compute_mean_std(split_data["train"]["data"], bands)
            mean = stats["mean"]
            std = stats["std"]

            transformations[split_name] = {
                "train": EuroSATTransform(mean=mean, std=std, augment=True),
                "val_test": EuroSATTransform(mean=mean, std=std, augment=False),
            }

        # Create datasets and dataloaders
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Creating datasets and dataloaders...")
        dataloaders = {}
        for split_name, split_data in splits.items():
            datasets = {
                "train": EuroSATDataset(split_data["train"]["data"], split_data["train"]["labels"],
                                        transform=transformations[split_name]["train"]),
                "validation": EuroSATDataset(split_data["validation"]["data"], split_data["validation"]["labels"],
                                            transform=transformations[split_name]["val_test"]),
                "test": EuroSATDataset(split_data["test"]["data"], split_data["test"]["labels"],
                                    transform=transformations[split_name]["val_test"]),
            }
            dataloaders[split_name] = utils.create_dataloaders(datasets, batch_size=32)

        # Process each split
        results = {}
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        for split_name in splits.keys():
            print(f"\nProcessing {split_name} split...")
            split_output_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)
            results[split_name] = process_split(
                split_name,
                dataloaders[split_name],
                device,
                output_dir,
                save_dir,
                load_saved_models,
                class_map  # Pass class_map to process_split
            )
            metrics.plot_metrics(
                train_losses=results[split_name]["train_losses"],
                val_losses=results[split_name]["val_losses"],
                train_accuracies=results[split_name]["train_accuracies"],
                val_accuracies=results[split_name]["val_accuracies"],
                test_accuracy=results[split_name]["test_accuracy"],
                output_dir=split_output_dir,
            )

        print("All splits processed. Results saved in the 'output' directory.")


def process_single_mat(file_path, save_dir, load_saved_models, no_valid, label_txt):
    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]

    print(f"Processing file: {file_path}")
    train_data, test_data, labels = utils.load_mat_data(file_path, label_txt)

    num_channels = train_data.shape[1]
    print(f"Number of channels: {num_channels}")

    mean, std = utils.compute_mean_std_mat(train_data)
    labels -= 1  # Convert MATLAB labels to zero-based index

    unique_labels = np.unique(labels)

    if no_valid:
        train_images, test_images = train_data, test_data
        normalize_transform = EuroSATNormalize(mean, std)
        datasets = {
            "train": EuroSATMatDataset(train_images, labels, transform=normalize_transform),
            "test": EuroSATMatDataset(test_images, transform=normalize_transform),
        }
        batch_size = 32
        dataloaders = {
            "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, pin_memory=True),
            "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, pin_memory=True),
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(save_dir, f"model_{file_name_without_ext}.pth")
        metrics_path = os.path.join(save_dir, f"metrics_{file_name_without_ext}.pkl")

        model = AdaptedHybridModel(num_channels=num_channels, num_classes=len(unique_labels)).to(device)
        optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        # optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-2)
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, epochs=25,
        #                                           steps_per_epoch=len(dataloaders["train"]))
        criterion = nn.CrossEntropyLoss()

        if load_saved_models and os.path.exists(model_path):
            print(f"Loading saved model for {file_name_without_ext} ...")
            model.load_state_dict(torch.load(model_path))
            metrics_loaded = utils.load_metrics(metrics_path)
        else:
            print(f"Training model for {file_name_without_ext} ...")
            metrics_computed = train_only.train_model(model, dataloaders["train"], criterion, optimizer, scheduler,
                                                      device, num_epochs=25)
            utils.save_metrics(metrics_computed, metrics_path)
            torch.save(model.state_dict(), model_path)

        print(f"Testing model for {file_name_without_ext} ...")
        test_logits.test_model(model, dataloaders["test"], device, save_dir, file_name_without_ext)

    print(f"Processing for {file_name_without_ext} completed.")


def execute_dir(dataset_dir, save_dir, load_saved_models, no_valid, labels_txt):
    mat_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".mat")])
    for file_name in mat_files:
        if file_name.endswith(".mat"):
            file_path = os.path.join(dataset_dir, file_name)
            process_single_mat(file_path, save_dir, load_saved_models, no_valid, labels_txt)
            print(f"Finished processing {file_name}.")


if __name__ == '__main__':
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    os.makedirs(parsed_args.saves_dir, exist_ok=True)
    if parsed_args.directory:
        execute_dir(parsed_args.dataset_path, parsed_args.saves_dir, parsed_args.load_saved_models,
                    parsed_args.no_valid, parsed_args.labels_txt)
    else:
        execute(parsed_args.dataset_path, parsed_args.saves_dir, parsed_args.load_saved_models,
                parsed_args.compressed_dataset, parsed_args.mat_format, parsed_args.labels_txt,
                parsed_args.no_valid)
