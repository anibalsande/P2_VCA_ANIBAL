import torch
import torch.optim as optim


def train_experiment(model, train_loader, num_epochs, lr, device,
                     loss_fn, label_smoothing=0.0):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        correct = 0
        n_pixels = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            target = (masks * (1.0 - label_smoothing) + 0.5 * label_smoothing
                      if label_smoothing > 0.0 else masks)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                binary = (masks > 0.5).float()
                correct += (preds == binary).sum().item()
                n_pixels += binary.numel()

        epoch_loss = running_loss / n_samples
        epoch_acc = correct / n_pixels
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print(f"  Época [{epoch+1:3d}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    return losses, accuracies
