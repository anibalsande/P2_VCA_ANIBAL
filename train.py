import torch
import torch.optim as optim


def train_one_experiment(model, train_loader, num_epochs, lr, device,
                         loss_fn, label_smoothing=0.0, log_every=10):
    """
    Entrena la U-Net con LR fijo (sin scheduler).

    loss_fn        — callable(logits, target) -> scalar
    label_smoothing — suaviza los targets binarios: [0,1] → [ε/2, 1-ε/2]
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_samples    = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            if label_smoothing > 0.0:
                target = masks * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                target = masks

            optimizer.zero_grad()
            loss = loss_fn(model(images), target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_samples    += images.size(0)

        epoch_loss = running_loss / n_samples
        losses.append(epoch_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  Época [{epoch+1:3d}/{num_epochs}]  Loss: {epoch_loss:.4f}")

    return losses
