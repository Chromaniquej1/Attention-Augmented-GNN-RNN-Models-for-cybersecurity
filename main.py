import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import GATRNNWithAttentionModel
from train import train_model
from data_preprocess import load_data, preprocess_data
from model_evaluation import evaluate_model
from data_parser import parse_args
import argparse

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
args = parse_args()

# Load and preprocess data
data_path = args.data_path  # This is the path passed from the terminal
features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
label = 'label'
X, y = load_data(data_path, features, label)
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(X, y)

# Create edge index for graph structure
num_nodes_train = X_train_tensor.shape[0]
train_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_train - 1)]).t().contiguous()

num_nodes_test = X_test_tensor.shape[0]
test_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_test - 1)]).t().contiguous()

# Create train_data and test_data objects
train_data = Data(x=X_train_tensor, edge_index=train_edge_index, y=y_train_tensor)
test_data = Data(x=X_test_tensor, edge_index=test_edge_index, y=y_test_tensor)

# Create DataLoader for train and test sets
train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
test_loader = DataLoader([test_data], batch_size=32, shuffle=False)

# Initialize the model
model = GATRNNWithAttentionModel(in_channels=X_train_tensor.shape[1], hidden_channels=64, rnn_hidden_size=32, out_channels=2)
model = model.to(device)

# Set the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, device, epochs=10)

# Evaluate the model
evaluate_model(model, test_loader, device)
