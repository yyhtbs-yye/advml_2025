import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_norm=False):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.use_norm = use_norm
        
        if self.use_norm:
            self.norm_i = nn.LayerNorm(hidden_size)
            self.norm_f = nn.LayerNorm(hidden_size)
            self.norm_g = nn.LayerNorm(hidden_size)
            self.norm_o = nn.LayerNorm(hidden_size)
        
        # Input gate weights
        self.W_i = nn.Linear(input_size, hidden_size)  # Input-to-input gate
        self.U_i = nn.Linear(hidden_size, hidden_size)  # Hidden-to-input gate
        
        # Forget gate weights
        self.W_f = nn.Linear(input_size, hidden_size)  # Input-to-forget gate
        self.U_f = nn.Linear(hidden_size, hidden_size)  # Hidden-to-forget gate
        
        # Cell state weights
        self.W_g = nn.Linear(input_size, hidden_size)  # Input-to-cell state
        self.U_g = nn.Linear(hidden_size, hidden_size)  # Hidden-to-cell state
        
        # Output gate weights
        self.W_o = nn.Linear(input_size, hidden_size)  # Input-to-output gate
        self.U_o = nn.Linear(hidden_size, hidden_size)  # Hidden-to-output gate

    def forward(self, x_t, h_c_prev):
        h_prev, c_prev = h_c_prev
        
        # Input gate
        i_t = self.W_i(x_t) + self.U_i(h_prev)
        i_t = self.norm_i(i_t) if self.use_norm else i_t
        i_t = torch.sigmoid(i_t)
        
        # Forget gate
        f_t = self.W_f(x_t) + self.U_f(h_prev)
        f_t = self.norm_f(f_t) if self.use_norm else f_t
        f_t = torch.sigmoid(f_t)
        
        # Cell update
        g_t = self.W_g(x_t) + self.U_g(h_prev)
        g_t = self.norm_g(g_t) if self.use_norm else g_t
        g_t = torch.tanh(g_t)
        
        # Output gate
        o_t = self.W_o(x_t) + self.U_o(h_prev)
        o_t = self.norm_o(o_t) if self.use_norm else o_t
        o_t = torch.sigmoid(o_t)
        
        # Cell state update
        c_t = f_t * c_prev + i_t * g_t
        
        # Hidden state update
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, X, h_c_0=None):
        B, T, _ = X.shape
        
        if h_c_0 is None:
            h_0 = torch.zeros(B, self.hidden_size).to(X.device)
            c_0 = torch.zeros(B, self.hidden_size).to(X.device)
            h_c_0 = (h_0, c_0)
        
        h_t, c_t = h_c_0
        H = []  # output of all hidden states
        
        for t in range(T):
            h_t, c_t = self.cell(X[:,t,:], (h_t, c_t))
            H.append(h_t)
        
        # Stack all hidden states
        H = torch.stack(H, dim=1)  # Shape: [B, T, D]
        
        return H, (h_t, c_t)

class LSTMBackbone(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMBackbone, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.layers = nn.ModuleList([LSTMLayer(input_size, hidden_size)])
        self.layers.extend([
            LSTMLayer(hidden_size, hidden_size) for _ in range(1, num_layers)])

    def forward(self, X, HC_0=None):
        """
        X: Input sequence (batch_size, seq_length, input_size)
        HC_0: Initial hidden and cell states tuple ((num_layers, batch_size, hidden_size), 
                                                   (num_layers, batch_size, hidden_size)) or None
        Returns:
            H_list: List of hidden states for each layer [layer][batch_size, seq_length, hidden_size]
            HC_n: Tuple of final hidden and cell states for each layer
        """
        B, T, _ = X.shape
        
        if HC_0 is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(X.device)
            c_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(X.device)
            HC_0 = (h_0, c_0)
        
        h_0, c_0 = HC_0
        
        H_list = []
        h_n_list = []
        c_n_list = []
        
        for l, lstm_layer in enumerate(self.layers):
            H, (h_n, c_n) = lstm_layer(X, (h_0[l], c_0[l]))
            X = H  # Output of this layer becomes input to the next
            
            H_list.append(H)
            h_n_list.append(h_n)
            c_n_list.append(c_n)
        
        # Stack final hidden and cell states
        h_n = torch.stack(h_n_list, dim=0)  # Shape: [num_layers, B, D]
        c_n = torch.stack(c_n_list, dim=0)  # Shape: [num_layers, B, D]
        
        return H_list, (h_n, c_n)