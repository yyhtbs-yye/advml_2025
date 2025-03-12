import torch
import torch.nn as nn

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        A single LSTM cell with separated gate computations.
        Gates:
          - Input gate (i)
          - Forget gate (f)
          - Output gate (o)
          - Candidate (g): often called the cell gate (not passed through a sigmoid)
        """
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate parameters
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forget gate parameters
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output gate parameters
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        # Candidate cell state (sometimes called the "cell gate")
        self.W_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    def init_weights(self):
        # Initialize weights uniformly (you can choose another initializer)
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
            
    def forward(self, x, hidden):
        """
        x: (batch_size, input_size)
        hidden: tuple of (h_prev, c_prev) each of shape (batch_size, hidden_size)
        """
        h_prev, c_prev = hidden
        
        # Compute each gate separately
        i = torch.sigmoid(x @ self.W_i + h_prev @ self.U_i + self.b_i)  # Input gate
        f = torch.sigmoid(x @ self.W_f + h_prev @ self.U_f + self.b_f)  # Forget gate
        o = torch.sigmoid(x @ self.W_o + h_prev @ self.U_o + self.b_o)  # Output gate
        g = torch.tanh(x @ self.W_c + h_prev @ self.U_c + self.b_c)       # Candidate cell state
        
        # Update cell state and hidden state
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        A multi-layer LSTM that uses MyLSTMCell to iterate through a sequence.
        x shape: (seq_len, batch_size, input_size)
        """
        super(MyLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Create a ModuleList of LSTM cells for each layer.
        self.cells = nn.ModuleList(
            [MyLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
             for i in range(num_layers)]
        )

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            # Initialize hidden states for all layers (h, c)
            hidden = [(torch.zeros(batch_size, self.hidden_size, device=x.device),
                       torch.zeros(batch_size, self.hidden_size, device=x.device))
                      for _ in range(self.num_layers)]
        
        outputs = []
        # Process the sequence one time step at a time.
        for t in range(seq_len):
            input_t = x[t]
            for layer in range(self.num_layers):
                h, c = self.cells[layer](input_t, hidden[layer])
                hidden[layer] = (h, c)
                input_t = h  # the output of the current layer becomes input to the next
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)  # shape: (seq_len, batch_size, hidden_size)
        return outputs, hidden
