import os
import matplotlib.pyplot as plt
import tifffile as tiff
import torch


def load_data_paths(data_dir):
    # This function loads image paths and corresponding labels
    image_paths = []
    labels = []
    class_map = {}

    classes = sorted(os.listdir(data_dir))
    for idx, class_name in enumerate(classes):
        class_map[class_name] = idx
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.endswith(".tif"):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(idx)

    return image_paths, labels, class_map


def load_image_torch(image_path):
    # Load a 13-channel image using torchvision's read_image
    image = tiff.imread(image_path)
    image = torch.from_numpy(image).float()
    # # Ensure the image has shape [channels, height, width]
    # if len(image.shape) == 2:  # Case where image is [height, width]
    #     image = image.unsqueeze(0)  # Add a channel dimension
    return image


# Function to split the data into train, validation, and test sets
def split_data(image_paths, labels, train_split=0.8, validation_split=0.1):
    # Pair the image paths and labels together
    data = list(zip(image_paths, labels))

    # Sort by label to ensure a balanced split
    data = sorted(data, key=lambda x: x[1])

    # Calculate sizes of each split
    total_size = len(data)
    train_size = int(train_split * total_size)
    val_size = int(validation_split * total_size)

    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Unzip the data back into separate image paths and labels
    train_image_paths, train_labels = zip(*train_data)
    val_image_paths, val_labels = zip(*val_data)
    test_image_paths, test_labels = zip(*test_data)

    # Return the lists for each split
    return list(train_image_paths), list(train_labels), list(val_image_paths), list(val_labels), list(
        test_image_paths), list(test_labels)


metrics_names = ['Accuracy', 'Kappa', 'Precision', 'Recall', 'F1-score']


def plot_metrics(metrics_dict, title):
    plt.figure(figsize=(12, 6))
    for idx, (model, metrics) in enumerate(metrics_dict.items()):
        plt.bar([x + idx * 0.2 for x in range(len(metrics))], metrics, width=0.2, label=model)
    plt.xticks([r + 0.2 for r in range(len(metrics_names))], metrics_names)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the training and validation losses and accuracies over epochs.

    Args:
    - train_losses (list): List of training loss values for each epoch.
    - val_losses (list): List of validation loss values for each epoch.
    - train_accuracies (list): List of training accuracy values for each epoch.
    - val_accuracies (list): List of validation accuracy values for each epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

