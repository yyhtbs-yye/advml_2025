import torch
import torch.nn as nn

class SpeechRecognizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.ln = nn.LayerNorm(2*hidden_dim)               # LayerNorm with correct dimension
        self.dropout_lstm = nn.Dropout(dropout_rate)    # Dropout after LSTM
        self.fc   = nn.Linear(hidden_dim*2, vocab_size)  # output layer for CTC (includes blank)
    def forward(self, x):
        # x.shape: (T, input_dim)
        o, _ = self.lstm(x)                # (T, 1, 2*hidden_dim)
        o = self.ln(o)
        o = self.dropout_lstm(o)          # Apply dropout to LSTM output
        o = self.fc(o)                   # (T, 1, vocab_size) logits for each timestep
        log_probs = nn.functional.log_softmax(o, dim=2)  # CTC requires log-probs
        return log_probs  # shape (T, 1, vocab_size)
