import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    Train the given model with specified hyperparameters.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        scheduler (torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau): Learning rate scheduler.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        tuple: A tuple containing four lists:
            - Training losses for each epoch.
            - Validation losses for each epoch.
            - Training accuracies for each epoch.
            - Validation accuracies for each epoch.
    """
    model.to(device)
    scaler = GradScaler()

    # Initialize lists to store losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

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

                # Mixed precision training
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Scale loss and update weights
                scaler.scale(loss).backward()
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
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            with tqdm(validation_loader, unit="batch") as t_val:
                t_val.set_description(f"Epoch {epoch + 1}/{num_epochs} - Validation")
                for inputs, labels in t_val:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Mixed precision evaluation
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    running_val_loss += loss.item()

                    # Compute validation accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

                    t_val.set_postfix(loss=running_val_loss / (t_val.n + 1))

        # Compute average validation loss and accuracy
        epoch_val_loss = running_val_loss / len(validation_loader)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        # Log epoch results
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
            f"Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}"
        )

    return train_losses, val_losses, train_accuracies, val_accuracies
