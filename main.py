import argparse
import sys
import os
from utils import utils
from base.EuroSATDataset import EuroSATDataset
from transform.EuroSATTransform import EuroSATTransform
from model.CustomCNN import CustomCNN
from model.VisionTransformer import VisionTransformer
from model.EnsembleModel import EnsembleModel
from trainer import train, evaluate
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet50_Weights, GoogLeNet_Weights
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_arg_parser():
    # Creates and returns the ArgumentParser object
    parser = argparse.ArgumentParser(description='Land Cover Classification using Sentinel-2 data.')
    parser.add_argument('datasetPath',
                    help='Path to the dataset directory.')
    return parser


def execute(data_dir):
    image_paths, labels, class_map = utils.load_data_paths(data_dir)

    # Means and standard deviations for the 13 spectral bands of the EuroSAT dataset
    mean = [1004.0, 1183.0, 1043.0, 952.0, 1244.0, 1678.0, 2193.0, 2260.0, 2088.0, 1595.0, 1313.0, 1165.0, 870.0]
    std = [507.0, 457.0, 443.0, 457.0, 515.0, 733.0, 1098.0, 1199.0, 1098.0, 923.0, 782.0, 690.0, 463.0]

    # DataLoader Parameters
    batch_size = 32

    # Check if CUDA is available
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the 80-20 split with 10% of training for validation
    train_image_paths_80, train_labels_80, validation_image_paths_80, validation_labels_80, test_image_paths_80, test_labels_80 = utils.split_data(
        image_paths, labels, train_split=0.8, validation_split=0.1)

    # Create the 50-50 split with 10% of training for validation
    train_image_paths_50, train_labels_50, validation_image_paths_50, validation_labels_50, test_image_paths_50, test_labels_50 = utils.split_data(
        image_paths, labels, train_split=0.5, validation_split=0.1)

    # Create Tranform Instances for both splits
    train_transform = EuroSATTransform(mean=mean, std=std, augment=True)
    validation_test_transform = EuroSATTransform(mean=mean, std=std, augment=False)

    # Create Dataset Instances
    train_dataset_80 = EuroSATDataset(train_image_paths_80, train_labels_80, transform=train_transform)
    validation_dataset_80 = EuroSATDataset(validation_image_paths_80, validation_labels_80, transform=validation_test_transform)
    test_dataset_80 = EuroSATDataset(test_image_paths_80, test_labels_80, transform=validation_test_transform)

    train_dataset_50 = EuroSATDataset(train_image_paths_50, train_labels_50, transform=train_transform)
    validation_dataset_50 = EuroSATDataset(validation_image_paths_50, validation_labels_50, transform=validation_test_transform)
    test_dataset_50 = EuroSATDataset(test_image_paths_50, test_labels_50, transform=validation_test_transform)

    # Create the Data Loaders for 80-20 split
    train_loader_80 = DataLoader(train_dataset_80, batch_size=batch_size, shuffle=True)
    validation_loader_80 = DataLoader(validation_dataset_80, batch_size=batch_size, shuffle=False)
    test_loader_80 = DataLoader(test_dataset_80, batch_size=batch_size, shuffle=False)

    # Create the Data Loaders for 50-50 split
    train_loader_50 = DataLoader(train_dataset_50, batch_size=batch_size, shuffle=True)
    validation_loader_50 = DataLoader(validation_dataset_50, batch_size=batch_size, shuffle=False)
    test_loader_50 = DataLoader(test_dataset_50, batch_size=batch_size, shuffle=False)

    # Instantiate models
    cnn_model = CustomCNN(num_classes=10)
    vit_model = VisionTransformer(num_classes=10)
    ensemble_model = EnsembleModel(cnn_model, vit_model, num_classes=10)

    # # Load pretrained ResNet50 and GoogleNet models
    # resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # resnet50.fc = nn.Linear(resnet50.fc.in_features, 10)
    #
    # googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    # googlenet.fc = nn.Linear(googlenet.fc.in_features, 10)

    # Hyperparameters
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()

    print("Training and evaluating for 80-20 split...")
    # Ensemble Model (80-20 split)
    optimizer_ensemble = optim.Adam(ensemble_model.parameters(), lr=learning_rate)
    scheduler_ensemble = ReduceLROnPlateau(optimizer_ensemble, 'min', patience=3, factor=0.5)
    train_losses_ensemble_80, val_losses_ensemble_80, train_accuracies_ensemble_80, val_accuracies_ensemble_80 = train.train_model(ensemble_model, train_loader_80, validation_loader_80, criterion, optimizer_ensemble, scheduler_ensemble, device, num_epochs=25)
    metrics_ensemble_80 = evaluate.evaluate_model(ensemble_model, test_loader_80, device)

    # # ResNet50 Model (80-20 split)
    # optimizer_resnet = optim.Adam(resnet50.parameters(), lr=learning_rate)
    # train_losses_resnet50_80, val_losses_resnet50_80, val_accuracies_resnet50_80 = train.train_model(resnet50, train_loader_80, validation_loader_80, criterion, optimizer_resnet, device, num_epochs=25)
    # metrics_resnet50_80 = evaluate.evaluate_model(resnet50, test_loader_80, device)
    #
    # # GoogleNet Model (80-20 split)
    # optimizer_googlenet = optim.Adam(googlenet.parameters(), lr=learning_rate)
    # train_losses_googlenet_80, val_losses_googlenet_80, val_accuracies_googlenet_80 = train.train_model(googlenet, train_loader_80, validation_loader_80, criterion, optimizer_googlenet, device, num_epochs=25)
    # metrics_googlenet_80 = evaluate.evaluate_model(googlenet, test_loader_80, device)

    print("Training and evaluating for 50-50 split...")
    # Ensemble Model (50-50 split)
    optimizer_ensemble = optim.Adam(ensemble_model.parameters(), lr=learning_rate)
    train_losses_ensemble_50, val_losses_ensemble_50, train_accuracies_ensemble_50, val_accuracies_ensemble_50 = train.train_model(ensemble_model, train_loader_50, validation_loader_50, criterion, optimizer_ensemble, scheduler_ensemble, device, num_epochs=25)
    metrics_ensemble_50 = evaluate.evaluate_model(ensemble_model, test_loader_50, device)

    # # ResNet50 Model (50-50 split)
    # optimizer_resnet = optim.Adam(resnet50.parameters(), lr=learning_rate)
    # train_losses_resnet50_50, val_losses_resnet50_50, val_accuracies_resnet50_50 = train.train_model(resnet50, train_loader_50, validation_loader_50, criterion, optimizer_resnet, device, num_epochs=25)
    # metrics_resnet50_50 = evaluate.evaluate_model(resnet50, test_loader_50, device)
    #
    # # GoogleNet Model (50-50 split)
    # optimizer_googlenet = optim.Adam(googlenet.parameters(), lr=learning_rate)
    # train_losses_googlenet_50, val_losses_googlenet_50, val_accuracies_googlenet_50 = train.train_model(googlenet, train_loader_50, validation_loader_50, criterion, optimizer_googlenet, device, num_epochs=25)
    # metrics_googlenet_50 = evaluate.evaluate_model(googlenet, test_loader_50, device)

    print("\n80-20 Split Metrics Comparison:")
    print("Custom Ensemble Model:", metrics_ensemble_80)
    # print("ResNet50 Model:", metrics_resnet50_80)
    # print("GoogleNet Model:", metrics_googlenet_80)

    print("\n50-50 Split Metrics Comparison:")
    print("Custom Ensemble Model:", metrics_ensemble_50)
    # print("ResNet50 Model:", metrics_resnet50_50)
    # print("GoogleNet Model:", metrics_googlenet_50)

    # Visualization for 80-20 split and 50-50 split
    metrics_80 = {
        'Ensemble': metrics_ensemble_80 #,
        # 'ResNet50': metrics_resnet50_80,
        # 'GoogleNet': metrics_googlenet_80
    }
    metrics_50 = {
        'Ensemble': metrics_ensemble_50 #,
        # 'ResNet50': metrics_resnet50_50,
        # 'GoogleNet': metrics_googlenet_50
    }

    utils.plot_metrics(metrics_80, "Model Performance on 80-20 Split")
    utils.plot_metrics(metrics_50, "Model Performance on 50-50 Split")

    utils.plot_training_curves(train_losses_ensemble_80, val_losses_ensemble_80, train_accuracies_ensemble_80, val_accuracies_ensemble_80,
                         "Ensemble Model Training (80-20 Split)")
    # utils.plot_training_curves(train_losses_resnet50_80, val_losses_resnet50_80, val_accuracies_resnet50_80,
    #                      "ResNet50 Model Training (80-20 Split)")
    # utils.plot_training_curves(train_losses_googlenet_80, val_losses_googlenet_80, val_accuracies_googlenet_80,
    #                      "GoogleNet Model Training (80-20 Split)")

    utils.plot_training_curves(train_losses_ensemble_50, val_losses_ensemble_50, train_accuracies_ensemble_50, val_accuracies_ensemble_50,
                         "Ensemble Model Training (50-50 Split)")
    # utils.plot_training_curves(train_losses_resnet50_50, val_losses_resnet50_50, val_accuracies_resnet50_50,
    #                      "ResNet50 Model Training (50-50 Split)")
    # utils.plot_training_curves(train_losses_googlenet_50, val_losses_googlenet_50, val_accuracies_googlenet_50,
    #                      "GoogleNet Model Training (50-50 Split)")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.basename(parsed_args.datasetPath):
        execute(parsed_args.datasetPath)

