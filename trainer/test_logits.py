import torch
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F


def test_model(model, test_loader, device, save_dir, file_name_without_ext):
    """
    Evaluate the model on the test dataset and save logits/probs.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for evaluation.
        save_dir (str): Path where to save the logits/probs.
        file_name_without_ext (str): Name of the input file.

    Returns:
        numpy.ndarray: Logits of the test set.
    """
    model.eval()
    model.to(device)

    # Store logits and probabilities
    all_logits = []
    all_probs = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', unit='batch')
        for inputs in test_bar:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)  # Logits (no softmax)
            probabilities = F.softmax(outputs, dim=1)

            # Convert to CPU and store
            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    print(f"all_logits shape: {all_logits.shape}")
    print(f"all_probs shape: {all_probs.shape}")

    # Save logits and probs
    save_path = os.path.join(save_dir, f"logits_{file_name_without_ext}.npy")
    np.save(save_path, all_logits)
    save_prob_path = os.path.join(save_dir, f"probs_{file_name_without_ext}.npy")
    np.save(save_prob_path, all_probs)

    return all_logits, all_probs
