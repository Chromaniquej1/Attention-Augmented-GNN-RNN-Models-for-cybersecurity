import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader


n_epochs = 100
def train_model(model, train_loader, criterion, optimizer, device, epochs=n_epochs):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
