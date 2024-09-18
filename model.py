import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs, mask=None):
        attn_weights = torch.matmul(encoder_outputs, self.v)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=1)
        return context, attn_weights

class GATRNNWithAttentionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, rnn_hidden_size, out_channels):
        super(GATRNNWithAttentionModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.rnn = nn.LSTM(hidden_channels * 2, rnn_hidden_size, batch_first=True)
        self.attention = Attention(rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = x.view(len(data.y), 1, -1)
        lstm_out, (hn, _) = self.rnn(x)
        context, attn_weights = self.attention(hn[-1], lstm_out, lstm_out)
        out = self.fc(context)
        return out
