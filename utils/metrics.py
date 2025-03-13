import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
    for i, loss in enumerate(train_losses):
        if i == 0 or i == len(train_losses) - 1 or i % (len(train_losses) // 4) == 0:
            plt.text(i, loss, f"{loss:.2f}", fontsize=9, ha='right', color='blue')
    for i, loss in enumerate(val_losses):
        if i == 0 or i == len(val_losses) - 1 or i % (len(val_losses) // 4) == 0:
            plt.text(i, loss, f"{loss:.2f}", fontsize=9, ha='left', color='orange')
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
    for i, acc in enumerate(train_accuracies):
        if i == 0 or i == len(train_accuracies) - 1 or i % (len(train_accuracies) // 4) == 0:
            plt.text(i, acc, f"{acc*100:.2f}%", fontsize=9, ha='right', color='blue')
    for i, acc in enumerate(val_accuracies):
        if i == 0 or i == len(val_accuracies) - 1 or i % (len(val_accuracies) // 4) == 0:
            plt.text(i, acc, f"{acc*100:.2f}%", fontsize=9, ha='left', color='orange')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    acc_plot_path = os.path.join(output_dir, "accuracy_per_epoch.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy plot saved at: {acc_plot_path}")

    # Compare the test accuracy with the training and validation accuracy of the last epoch
    plt.figure(figsize=(10, 6))
    accuracies = [train_accuracies[-1] * 100, val_accuracies[-1] * 100, test_accuracy * 100]
    labels = ["Train Accuracy", "Validation Accuracy", "Test Accuracy"]
    bars = plt.bar(labels, accuracies, color=["skyblue", "orange", "green"])
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f"{acc:.2f}%",
                 ha='center', va='bottom', fontsize=10, color='black')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    comparison_plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Accuracy comparison plot saved at: {comparison_plot_path}")


def plot_metrics_train(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """
    Plots training/validation losses, accuracies, and saves the plots.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Training and Validation Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    for i, loss in enumerate(train_losses):
        if i == 0 or i == len(train_losses) - 1 or i % (len(train_losses) // 4) == 0:
            plt.text(i, loss, f"{loss:.2f}", fontsize=9, ha='right', color='blue')
    for i, loss in enumerate(val_losses):
        if i == 0 or i == len(val_losses) - 1 or i % (len(val_losses) // 4) == 0:
            plt.text(i, loss, f"{loss:.2f}", fontsize=9, ha='left', color='orange')
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
    for i, acc in enumerate(train_accuracies):
        if i == 0 or i == len(train_accuracies) - 1 or i % (len(train_accuracies) // 4) == 0:
            plt.text(i, acc, f"{acc*100:.2f}%", fontsize=9, ha='right', color='blue')
    for i, acc in enumerate(val_accuracies):
        if i == 0 or i == len(val_accuracies) - 1 or i % (len(val_accuracies) // 4) == 0:
            plt.text(i, acc, f"{acc*100:.2f}%", fontsize=9, ha='left', color='orange')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    acc_plot_path = os.path.join(output_dir, "accuracy_per_epoch.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy plot saved at: {acc_plot_path}")


def plot_metrics_train_only(train_losses, train_accuracies, output_dir):
    """
    Plots training/validation losses, accuracies, and saves the plots.

    Args:
        train_losses (list): List of training losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Training and Validation Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    for i, loss in enumerate(train_losses):
        if i == 0 or i == len(train_losses) - 1 or i % (len(train_losses) // 4) == 0:
            plt.text(i, loss, f"{loss:.2f}", fontsize=9, ha='right', color='blue')
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
    for i, acc in enumerate(train_accuracies):
        if i == 0 or i == len(train_accuracies) - 1 or i % (len(train_accuracies) // 4) == 0:
            plt.text(i, acc, f"{acc*100:.2f}%", fontsize=9, ha='right', color='blue')

    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    acc_plot_path = os.path.join(output_dir, "accuracy_per_epoch.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy plot saved at: {acc_plot_path}")


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

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Normalized Confusion Matrix', fontsize=16)

    # Set tick marks and labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12, ha="right")
    plt.yticks(tick_marks, classes, fontsize=12)

    # Add annotations for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            plt.text(j, i, value, horizontalalignment="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)

    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    # Save the plot
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