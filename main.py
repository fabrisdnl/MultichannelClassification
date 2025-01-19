import argparse
import sys
import os
from utils import utils, metrics
from base.EuroSATDataset import EuroSATDataset
from transform.EuroSATTransform import EuroSATTransform
from model.HybridModel import HybridModel
from trainer import train, test
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
    return parser


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


def execute(data_dir, save_dir, load_saved_models):
    # Load all images into memory
    print("Loading images into memory...")
    all_images, all_labels, class_map = utils.load_all_images(data_dir)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    execute(parsed_args.dataset_path, parsed_args.saves_dir, parsed_args.load_saved_models)
