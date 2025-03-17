import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler


def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    Train the given model without a validation phase.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        scheduler (torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau): Learning rate scheduler.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        metrics: A dict containing two lists:
            - Training losses for each epoch.
            - Training accuracies for each epoch.
    """
    # model.apply(initialize_weights)

    model.to(device)
    scaler = GradScaler('cuda')

    print(f"Using device: {device}")
    print(f"Model on GPU? {next(model.parameters()).is_cuda}")

    # Initialize lists to store losses and accuracies
    metrics = {'train_losses': [], 'train_accuracies': []}

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        with tqdm(train_loader, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch + 1}/{num_epochs} - Training")
            for inputs, labels in t_epoch:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"[DEBUG] NaN o Inf trovati nei dati di input! Batch index: {t_epoch.n}")
                    print(
                        f"Min: {inputs.min().item()}, Max: {inputs.max().item()}, Mean: {inputs.mean().item()}, Std: {inputs.std().item()}")

                # Mixed precision training
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    # print(f"[DEBUG] Model output min: {outputs.min().item()}, max: {outputs.max().item()}")
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"[DEBUG] NaN o Inf nei logits all'epoch {epoch + 1}, batch {t_epoch.n}")
                        print(
                            f"Logits min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")
                    loss = criterion(outputs, labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[DEBUG] Loss è NaN o Inf all'epoch {epoch + 1}, batch {t_epoch.n}")
                        print(f"Loss value: {loss.item()}")

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        max_grad = param.grad.abs().max()
                        print(f"[DEBUG] Gradiente max per {name}: {max_grad}")

                # Scale loss and update weights
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                scaler.step(optimizer)
                scaler.update()

                running_train_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

                t_epoch.set_postfix(loss=running_train_loss / (t_epoch.n + 1))

        # Compute average training loss and accuracy
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        metrics['train_losses'].append(epoch_train_loss)
        metrics['train_accuracies'].append(epoch_train_accuracy)

        # Scheduler step
        scheduler.step(epoch_train_loss)

        # Log epoch results
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Train Accuracy: {epoch_train_accuracy:.4f}"
        )

    return metrics
