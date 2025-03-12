import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout_embed = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout_lstm = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)  # 2 sentiment classes
        self.init_weights() 
        
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    

    def forward(self, x, lengths):
        # x: (batch, seq_len) of word indices
        embeds = self.embed(x)                  # (batch, seq_len, embed_dim)
        embeds = self.dropout_embed(embeds)     # Apply dropout to embeddings
        
        # Pack the sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Run through LSTM
        _, (h_n, c_n) = self.lstm(packed)  # h_n, c_n: (1, batch, hidden_dim)
        
        # Squeeze the first dimension (num_layers * num_directions)
        h_final = h_n.squeeze(0)  # (batch, hidden_dim)
        # c_final = c_n.squeeze(0)  # (batch, hidden_dim)
        
        # Concatenate hidden and cell states
        # combined = torch.cat([h_final, c_final], dim=1)  # (batch, hidden_dim*2)
        h_final = self.dropout_lstm(h_final)  # Apply dropout before final layer

        h_final = self.ln(h_final)

        # Apply final classification layer
        out = self.fc(h_final)  # (batch, 2)
        return out