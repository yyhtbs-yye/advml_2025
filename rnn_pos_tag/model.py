import torch 
import torch.nn as nn

class POSTagger(nn.Module):
    def __init__(self, vocab_size, tag_count, emb_dim=100, hidden_dim=64, dropout_rate=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout_emb = nn.Dropout(dropout_rate)     # Dropout after embeddings
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout_lstm = nn.Dropout(dropout_rate)    # Dropout after LSTM
        self.ln = nn.LayerNorm(2*hidden_dim)               # LayerNorm with correct dimension
        self.fc = nn.Linear(2*hidden_dim, tag_count)       # Correct dimension since bidirectional=False
        
    def forward(self, word_idx_seq):
        embeds = self.embed(word_idx_seq)               
        embeds = self.dropout_emb(embeds)               # Apply dropout to embeddings
        lstm_out, _ = self.lstm(embeds)               
        lstm_out = self.ln(lstm_out)
        lstm_out = self.dropout_lstm(lstm_out)          # Apply dropout to LSTM output
        outputs = self.fc(lstm_out)                   
        return outputs