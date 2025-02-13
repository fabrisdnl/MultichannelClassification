import torch
from tqdm import tqdm
import numpy as np

def test_model(model, test_loader, device, save_dir):
    """
    Evaluate the model on the test dataset and save logits.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for evaluation.
        save_path (str): Path where to save the logits.

    Returns:
        numpy.ndarray: Logits of the test set.
    """
    model.eval()
    model.to(device)

    # Store logits
    all_logits = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', unit='batch')
        for inputs in test_bar:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)  # Logits (no softmax)

            # Convert to CPU and store
            all_logits.append(outputs.cpu().numpy())

    # Concatenate all logits
    all_logits = np.concatenate(all_logits, axis=0)

    # Save logits to a file
    save_path = os.path.join(save_dir, f"logits.npy")
    np.save(save_path, all_logits)

    return all_logits
