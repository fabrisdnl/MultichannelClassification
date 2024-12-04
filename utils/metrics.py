import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy, output_dir):
    """
    Plots training/validation losses, accuracies, and test accuracy, and saves the plots.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
        test_accuracy (float): Final accuracy on the test dataset.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Training and Validation Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    loss_plot_path = os.path.join(output_dir, "loss_per_epoch.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved at: {loss_plot_path}")

    # Plot Training and Validation Accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    acc_plot_path = os.path.join(output_dir, "accuracy_per_epoch.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy plot saved at: {acc_plot_path}")

    # Plot Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(["Test Accuracy"], [test_accuracy * 100], color="skyblue")
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    test_acc_plot_path = os.path.join(output_dir, "test_accuracy.png")
    plt.savefig(test_acc_plot_path)
    plt.close()
    print(f"Test accuracy plot saved at: {test_acc_plot_path}")


def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """
    Plots a confusion matrix and saves it to the specified directory.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        classes (list): List of class names.
        output_dir (str): Directory to save the confusion matrix plot.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix: each row is divided by the row sum
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the figure
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Normalized Confusion Matrix')

    # Set tick marks and labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    # Add annotations for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            plt.text(j, i, value, horizontalalignment="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save the figure to the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved at: {output_path}")


def save_metrics(accuracy, output_dir):
    """
    Saves the accuracy metric to a text file in the specified directory.

    Args:
        accuracy (float): Accuracy of the model.
        output_dir (str): Directory to save the metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

    print(f"Metrics saved at: {metrics_path}")