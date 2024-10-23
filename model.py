import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class Attention(nn.Module):
    """
    Implements an attention mechanism that computes attention weights over the encoder outputs
    and generates a context vector by taking the weighted sum of the encoder outputs.

    Args:
        hidden_size (int): The size of the hidden layer (used for attention calculation).
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)  # Linear transformation for attention
        self.v = nn.Parameter(torch.rand(hidden_size))  # Learnable vector for attention calculation

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Forward pass for the attention mechanism.
        
        Args:
            hidden (torch.Tensor): The last hidden state from the RNN.
            encoder_outputs (torch.Tensor): The outputs of the encoder (LSTM outputs).
            mask (torch.Tensor, optional): Mask to ignore certain time steps (not used here).
        
        Returns:
            context (torch.Tensor): The context vector, which is a weighted sum of the encoder outputs.
            attn_weights (torch.Tensor): The attention weights applied over the encoder outputs.
        """
        # Compute attention weights (dot product of encoder outputs with vector v)
        attn_weights = torch.matmul(encoder_outputs, self.v)
        
        # Apply softmax to obtain normalized attention weights
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Compute context by applying the attention weights to the encoder outputs
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=1)
        
        return context, attn_weights

class GATRNNWithAttentionModel(nn.Module):
    """
    A hybrid model that combines Graph Attention Network (GAT) with an RNN and attention mechanism.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in the GAT layer.
        rnn_hidden_size (int): Size of the hidden state in the RNN.
        out_channels (int): Number of output classes (for classification).
    """
    def __init__(self, in_channels, hidden_channels, rnn_hidden_size, out_channels):
        super(GATRNNWithAttentionModel, self).__init__()
        # GAT layer: uses 2 attention heads, and the outputs are concatenated
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        
        # RNN (LSTM) layer: takes the GAT output as input
        self.rnn = nn.LSTM(hidden_channels * 2, rnn_hidden_size, batch_first=True)
        
        # Attention mechanism applied to the RNN hidden state
        self.attention = Attention(rnn_hidden_size)
        
        # Fully connected layer for final classification
        self.fc = nn.Linear(rnn_hidden_size, out_channels)

    def forward(self, data):
        """
        Forward pass of the model.
        
        Args:
            data (torch_geometric.data.Data): A PyTorch Geometric Data object containing features and graph structure.
        
        Returns:
            torch.Tensor: The model's output predictions.
        """
        # Extract node features and edge indices from the input data
        x, edge_index = data.x, data.edge_index
        
        # Apply GAT convolution to the node features
        x = self.gat1(x, edge_index)
        x = torch.relu(x)  # Apply ReLU activation
        
        # Reshape GAT output to feed into the LSTM (batch_size, sequence_length, feature_size)
        x = x.view(len(data.y), 1, -1)
        
        # Apply the LSTM (RNN) to the GAT outputs
        lstm_out, (hn, _) = self.rnn(x)
        
        # Apply the attention mechanism to the final hidden state of the LSTM
        context, attn_weights = self.attention(hn[-1], lstm_out, lstm_out)
        
        # Pass the context vector through the fully connected layer to get final predictions
        out = self.fc(context)
        
        return out
