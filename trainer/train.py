import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, num_epochs=25):
    model.to(device)
    scaler = GradScaler()  # For mixed precision training

    # Store metrics for plotting purposes
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training Phase
        with tqdm(train_loader, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch + 1}/{num_epochs} - Training")
            for inputs, labels in t_epoch:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():  # Mixed Precision
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_loss = running_loss / (t_epoch.n + 1)
                train_acc = 100. * correct / total

                # Update progress bar
                t_epoch.set_postfix(loss=train_loss, accuracy=train_acc)

        # Store train loss and accuracy for plotting
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation Phase (Evaluation during training)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(validation_loader, unit="batch") as t_val:
                t_val.set_description(f"Epoch {epoch + 1}/{num_epochs} - Validation")
                for inputs, labels in t_val:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast(device_type=device):  # Mixed Precision Evaluation
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

        val_loss /= len(validation_loader)
        val_acc = 100. * correct / total

        # Store validation loss and accuracy for plotting
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Scheduler step
        scheduler.step(val_loss)

        # Print validation metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies
