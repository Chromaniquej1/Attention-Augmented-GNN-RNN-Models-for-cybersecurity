import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader

n_epochs = 100  # Default number of epochs for training

def train_model(model, train_loader, criterion, optimizer, device, epochs=n_epochs):
    """
    Trains the model on the training dataset.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch_geometric.loader.DataLoader): DataLoader providing batches of training data.
        criterion (torch.nn.Module): The loss function to minimize (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam) for updating model parameters.
        device (torch.device): The device (CPU or GPU) on which to perform training.
        epochs (int): Number of training epochs (default is 100).
        
    Returns:
        None: Prints the loss for each epoch during training.
    """
    model.train()  # Set the model to training mode

    # Loop over epochs
    for epoch in range(epochs):
        # Loop over each batch of data in the training set
        for batch in train_loader:
            batch = batch.to(device)  # Move the batch to the selected device (GPU/CPU)

            optimizer.zero_grad()  # Zero the gradients from the previous step
            
            out = model(batch)  # Perform a forward pass through the model
            loss = criterion(out, batch.y)  # Compute the loss (compare predictions with true labels)
            
            loss.backward()  # Backpropagate the loss through the network
            optimizer.step()  # Update the model parameters using the optimizer

        # Print the loss for the current epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
