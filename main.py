import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import GATRNNWithAttentionModel  # Importing the custom GNN-RNN model with attention mechanism
from train import train_model  # Importing the training function
from data_preprocess import load_data, preprocess_data  # Importing data loading and preprocessing functions
from model_evaluation import evaluate_model  # Importing model evaluation function
from data_parser import parse_args  # Importing argument parsing function
import argparse

# Set the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments from the command line
args = parse_args()

# Load and preprocess data
data_path = args.data_path  # Path to the dataset file, passed from the terminal
features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']  # List of feature column names
label = 'label'  # Name of the target label column
X, y = load_data(data_path, features, label)  # Load the dataset based on provided features and label
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(X, y)  # Preprocess data for model

# Create edge index for the graph structure in the training data
num_nodes_train = X_train_tensor.shape[0]  # Number of nodes in the training data
train_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_train - 1)]).t().contiguous()  # Create edges

# Create edge index for the graph structure in the test data
num_nodes_test = X_test_tensor.shape[0]  # Number of nodes in the test data
test_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes_test - 1)]).t().contiguous()  # Create edges

# Create train_data and test_data objects using PyTorch Geometric's Data class
train_data = Data(x=X_train_tensor, edge_index=train_edge_index, y=y_train_tensor)  # Training data with features and edges
test_data = Data(x=X_test_tensor, edge_index=test_edge_index, y=y_test_tensor)  # Test data with features and edges

# Create DataLoader for training and test sets (with batch size and shuffle options)
train_loader = DataLoader([train_data], batch_size=32, shuffle=True)  # Loader for training data
test_loader = DataLoader([test_data], batch_size=32, shuffle=False)  # Loader for test data

# Initialize the GATRNNWithAttentionModel
model = GATRNNWithAttentionModel(
    in_channels=X_train_tensor.shape[1],  # Input feature dimension
    hidden_channels=64,  # Number of hidden units in the GNN layer
    rnn_hidden_size=32,  # Hidden size of the RNN layer
    out_channels=2  # Output dimension (2 classes for binary classification)
)
model = model.to(device)  # Move model to the selected device (GPU/CPU)

# Set the loss function (CrossEntropy for classification) and optimizer (Adam optimizer)
criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer with learning rate

# Train the model using the training data, criterion, optimizer, and device
train_model(model, train_loader, criterion, optimizer, device, epochs=10)

# Evaluate the model using the test data and the evaluation function
evaluate_model(model, test_loader, device)
