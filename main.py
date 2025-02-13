import argparse
import sys
import os
from utils import utils, metrics
from base.EuroSATDataset import EuroSATDataset
from base.EuroSATMatDataset import EuroSATMatDataset
from transform.EuroSATTransform import EuroSATTransform
from transform.EuroSATNormalize import EuroSATNormalize
from model.HybridModel import HybridModel
from trainer import train, test, test_logits
import torch
import torch.nn as nn
import torch.optim as optim
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
    return parser


def process_mat(dataloaders, device, output_dir, save_dir, load_saved_models, unique_labels, file_name_without_ext):
    """
    Process: trains and evaluates the model or loads a pre-trained model.

    Args:
        dataloaders (dict): Dataloaders for train, validation, and test.
        device (torch.device): Device to use for computations.
        save_dir (str): Directory where models will be saved.
        load_saved_models (bool): Whether to load pre-trained models.
        unique_labels (list): List of unique labels.
        file_name_without_ext (str): Name of the input file.

    Returns:
        dict: Dictionary containing metrics.
    """
    model_path = os.path.join(save_dir, f"model_{file_name_without_ext}.pth")
    metrics_path = os.path.join(save_dir, f"metrics_{file_name_without_ext}.pkl")
    model = HybridModel(num_classes=len(unique_labels)).to(device)  # Use the number of classes from class_map
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
    logits = test_logits.test_model(model, dataloaders["test"], device, save_dir)

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
    model = HybridModel(num_classes=len(class_map)).to(device)  # Use the number of classes from class_map
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


def execute(data_dir, save_dir, load_saved_models, compressed_dataset, mat_format):
    if mat_format:
        file_name = os.path.basename(data_dir)
        file_name_without_ext = os.path.splitext(file_name)[0]

        print(f"Loading MATLAB data from {data_dir}")
        train_data, test_data, labels = utils.load_mat_data(data_dir)

        # Compute mean and std on training images
        mean, std = utils.compute_mean_std_mat(train_images)

        # Normalization of labels: from [1-10] to [0,9]
        labels -= 1

        num_train = len(train_data)
        num_test = len(test_data)
        num_labels = len(labels)
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")
        print(f"Number of train+validation images loaded: {num_train}")
        print(f"Number of test images loaded: {num_test}")
        print(f"Number of labels loaded: {num_labels}")

        split_idx = int(num_train * 0.9)
        print(f"Index of validation set starting point: {split_idx}")
        train_images = train_data[:split_idx]  # 90% training
        valid_images = train_data[split_idx:]  # 10% validation
        test_images = test_data

        train_labels = labels[:split_idx]
        valid_labels = labels[split_idx:num_train]

        print(f"Train set: {train_images.shape}, Labels: {train_labels.shape}")
        print(f"Validation set: {valid_images.shape}, Labels: {valid_labels.shape}")
        print(f"Test set: {test_images.shape}")

        normalize_transform = EuroSATNormalize(mean, std)
        print("Creating datasets and dataloaders...")
        datasets = {
            "train": EuroSATMatDataset(train_images, train_labels, transform=normalize_transform),
            "validation": EuroSATMatDataset(valid_images, valid_labels, transform=normalize_transform),
            "test": EuroSATMatDataset(test_images, transform=normalize_transform),
        }
        dataloaders = utils.create_dataloaders(datasets, batch_size=32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        results = process_mat(dataloaders, device, output_dir, save_dir, load_saved_models, unique_labels, file_name_without_ext)

        metrics.plot_metrics_train(
            train_losses=results["train_losses"],
            val_losses=results["val_losses"],
            train_accuracies=results["train_accuracies"],
            val_accuracies=results["val_accuracies"],
            output_dir=split_output_dir,
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


if __name__ == '__main__':
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    os.makedirs(parsed_args.saves_dir, exist_ok=True)
    execute(parsed_args.dataset_path, parsed_args.saves_dir, parsed_args.load_saved_models,
            parsed_args.compressed_dataset, parsed_args.mat_format)
