import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, y_true, y_pred
