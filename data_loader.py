import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx

def data_preprocess_load(data_path):
    """
    Preprocesses the dataset and loads it for graph neural network training.

    Args:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        train_loader (DataLoader): PyTorch Geometric DataLoader for the training set.
        test_loader (DataLoader): PyTorch Geometric DataLoader for the test set.
    """
    
    # Load data from the CSV file
    df = pd.read_csv(data_path)

    # Select features and label from the dataset
    FEATURES = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']  # Feature columns for network traffic analysis
    LABEL = 'label'  # Assuming 'label' column represents attack/normal classification

    # Preprocess data: Scale the features and separate the label
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)  # Standardize the features
    y = df[LABEL].values  # Extract the labels

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)  # Training feature tensor
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)   # Training label tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)    # Test feature tensor
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)     # Test label tensor

    # Create a simple edge index to simulate graph structure (connect consecutive nodes)
    num_nodes_train = X_train.shape[0]  # Number of nodes in the training set
    train_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_train - 1)]).t().contiguous()

    num_nodes_test = X_test.shape[0]  # Number of nodes in the test set
    test_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_test - 1)]).t().contiguous()

    # Create PyTorch Geometric Data objects for the training and test sets
    train_data = Data(x=X_train_tensor, edge_index=train_edge_index, y=y_train_tensor)
    test_data = Data(x=X_test_tensor, edge_index=test_edge_index, y=y_test_tensor)

    # Create DataLoader for batching and shuffling the data
    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)  # DataLoader for training data
    test_loader = DataLoader([test_data], batch_size=32, shuffle=False)   # DataLoader for test data

    return train_loader, test_loader
