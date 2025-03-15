import torch
import torch.nn as nn
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_norm=False):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_size)

        # Weight matrices
        self.W = nn.Linear(input_size, hidden_size)  # Input-to-hidden
        self.U = nn.Linear(hidden_size, hidden_size) # Hidden-to-hidden

    def forward(self, x_t, h_prev):

        
        s = self.W(x_t) + self.U(h_prev)
        s = self.norm(s) if self.use_norm else s
        h_t = torch.tanh(s)
        return h_t

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, X, h_0=None):

        B, T, _ = X.shape

        if h_0 is None:
            h_0 = torch.zeros(B, self.hidden_size).to(X.device)
        
        h_t = h_0
        H = []
        for t in range(T):
            h_t = self.cell(X[:,t,:], h_t)
            H.append(h_t)
        
        # before: H.shape = [(T), (B, D)]

        H = torch.stack(H, dim=1)

        # after: H.shape = [B, T, D]

        return H

class RNNBackbone(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNBackbone, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.layers = nn.ModuleList([RNNLayer(input_size, hidden_size)])
        self.layers.extend([
            RNNLayer(hidden_size, hidden_size) for _ in range(1, num_layers)])           

    def forward(self, X, H_0=None):
        """
        X: Input sequence (batch_size, seq_length, input_size)
        H_0: Initial hidden state (num_layers, batch_size, hidden_size) or None
        Returns:
            H: List of hidden states for each layer [layer][batch_size, seq_length, hidden_size]
        """

        B, T, _ = X.shape

        if H_0 is None:
            H_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(X.device)

        H_list = []

        for l, rnn_layer in enumerate(self.layers):
            H = rnn_layer(X, H_0[l, :, :])
            X = H
            H_list.append(H)

        return H_list



