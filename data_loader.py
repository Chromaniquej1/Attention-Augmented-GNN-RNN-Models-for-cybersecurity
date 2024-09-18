import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx

def data_preprocess_load(data_path):

    df = pd.read_csv(data_path)

    # Select features and label
    FEATURES = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
    LABEL = 'label'  # Assuming 'label' column for attack/normal classification

    # Preprocess data: Scaling features and separating label
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)
    y = df[LABEL].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create a simple edge index for graph structure (connect consecutive nodes)
    num_nodes_train = X_train.shape[0]
    train_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_train - 1)]).t().contiguous()

    num_nodes_test = X_test.shape[0]
    test_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_test - 1)]).t().contiguous()

    # Create PyTorch Geometric graph data objects
    train_data = Data(x=X_train_tensor, edge_index=train_edge_index, y=y_train_tensor)
    test_data = Data(x=X_test_tensor, edge_index=test_edge_index, y=y_test_tensor)

    # Create DataLoader
    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
    test_loader = DataLoader([test_data], batch_size=32, shuffle=False)