import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        # Create a tqdm progress bar for the test_loader
        test_bar = tqdm(test_loader, desc='Evaluating', unit='batch')
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Update progress bar description
            test_bar.set_postfix({'Processed Batches': len(y_true) / len(test_loader.dataset)})

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, kappa, precision, recall, f1