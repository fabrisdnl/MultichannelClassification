import torch
from tqdm import tqdm
import numpy as np


def test_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for evaluation.

    Returns:
        tuple:
            - accuracy (float): Accuracy of the model on the test dataset.
            - y_pred (numpy.ndarray): Predicted labels.
            - y_true (numpy.ndarray): True labels.
    """
    model.eval()
    model.to(device)

    # Initiate lists to store labels
    y_true = []
    y_pred = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Evaluating', unit='batch')
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Collect predictions and true labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Update progress bar
            test_bar.set_postfix({'Batches Processed': len(y_true)})

    # Compute accuracy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)

    return accuracy, y_pred, y_true