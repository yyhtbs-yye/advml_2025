import torch
from torch import nn
from lstm import MyLSTMCell


class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        A bidirectional LSTM built from scratch.
        The forward and backward directions use independent sets of LSTM cells.
        The final output at each time step is the concatenation of both directions.
        """
        super(MyBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Forward LSTM layers
        self.forward_cells = nn.ModuleList(
            [MyLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
             for i in range(num_layers)]
        )
        # Backward LSTM layers
        self.backward_cells = nn.ModuleList(
            [MyLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
             for i in range(num_layers)]
        )

    def forward(self, x, hidden=None):
        """
        x shape: (seq_len, batch_size, input_size)
        Returns:
            outputs: (seq_len, batch_size, 2 * hidden_size)
            hidden: tuple of hidden states for forward and backward directions
        """
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            hidden_forward = [(torch.zeros(batch_size, self.hidden_size, device=x.device),
                                torch.zeros(batch_size, self.hidden_size, device=x.device))
                               for _ in range(self.num_layers)]
            hidden_backward = [(torch.zeros(batch_size, self.hidden_size, device=x.device),
                                 torch.zeros(batch_size, self.hidden_size, device=x.device))
                                for _ in range(self.num_layers)]
        else:
            hidden_forward, hidden_backward = hidden
        
        forward_outputs = []
        backward_outputs = []
        
        # Forward direction processing
        for t in range(seq_len):
            input_t = x[t]
            for layer in range(self.num_layers):
                h, c = self.forward_cells[layer](input_t, hidden_forward[layer])
                hidden_forward[layer] = (h, c)
                input_t = h
            forward_outputs.append(h.unsqueeze(0))
        
        # Backward direction processing (iterate backwards)
        for t in reversed(range(seq_len)):
            input_t = x[t]
            for layer in range(self.num_layers):
                h, c = self.backward_cells[layer](input_t, hidden_backward[layer])
                hidden_backward[layer] = (h, c)
                input_t = h
            # Insert at beginning so that the time order matches
            backward_outputs.insert(0, h.unsqueeze(0))
        
        forward_outputs = torch.cat(forward_outputs, dim=0)   # (seq_len, batch_size, hidden_size)
        backward_outputs = torch.cat(backward_outputs, dim=0) # (seq_len, batch_size, hidden_size)
        
        # Concatenate the outputs from both directions along the feature dimension.
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)  # (seq_len, batch_size, 2 * hidden_size)
        
        return outputs, (hidden_forward, hidden_backward)
