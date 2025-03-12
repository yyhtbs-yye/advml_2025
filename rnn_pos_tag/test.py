# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from helpers.ud_english_ewt_dataset import build_dataloader
from model import POSTagger

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# %%
BATCH_SIZE = 16
EPOCHS = 100

# %%
word_df = pd.read_csv("datasets/word_vocab.tsv", sep='\t')
word_vocab = {row['word']: row['index'] for _, row in word_df.iterrows()}

# Load tag vocabulary from tsv
tag_df = pd.read_csv("datasets/tag_vocab.tsv", sep='\t')
tag_vocab = {row['tag']: row['index'] for _, row in tag_df.iterrows()}

# %%
train_loader, val_loader = build_dataloader(
    "datasets/train_improved.tsv", 
    batch_size=BATCH_SIZE,
    split=True,
    shuffle=True,
    val_ratio=0.3
)

test_loader = build_dataloader(
    "datasets/test_improved.tsv",
    batch_size=BATCH_SIZE,
    split=False,
    shuffle=False,
)

# %%
device = 'cpu' # torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = POSTagger(vocab_size=len(word_vocab), tag_count=len(tag_vocab), emb_dim=32, hid_dim=32)
model.to(device)

# %%
criterion = nn.CrossEntropyLoss()  # tag classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# %%
def create_mask(lengths, max_len):
    mask = torch.zeros(len(lengths), max_len, device=lengths.device, dtype=torch.bool)
    for i, length in enumerate(lengths):
        # Clamp length to valid range
        valid_length = min(max(length.item(), 0), max_len)
        mask[i, :valid_length] = 1
    return mask


# %%
# 5. Train the model
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, targets, lengths in train_loader:
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(inputs)     # (seq_len, tag_count)

        B, T = inputs.size()
        # Create mask for real tokens (non-padding)
        mask = create_mask(lengths, T)
        
        # Apply mask to compute loss only on real tokens
        logits_masked = logits[mask]
        targets_masked = targets[mask]
        
        # Calculate loss on masked data
        loss = criterion(logits_masked, targets_masked)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: training loss = {avg_loss:.3f}")
    # Validate on validation set
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets, lengths in test_loader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            logits = model(inputs)

            B, T = inputs.size()

            # Create mask for real tokens (non-padding)
            mask = create_mask(lengths, T)
            
            # Apply mask to compute loss only on real tokens
            logits_masked = logits[mask]
            targets_masked = targets[mask]
            
            # Calculate loss on masked data
            loss = criterion(logits_masked, targets_masked)
            total_loss += loss.item() * B
            loss = total_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}: validation loss = {loss:.3f}")

# %%
# 6. Evaluate on test set
model.eval()
correct_tokens = 0
total_tokens = 0
with torch.no_grad():
    for inputs, targets, lengths in test_loader:
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        batch_size, max_len = inputs.size()
        
        # Forward pass (without passing lengths if not needed by model architecture)
        logits = model(inputs)  # Shape: [batch_size, max_len, num_tags]
        
        # Get predicted tags
        pred_tags = logits.argmax(dim=-1)  # Shape: [batch_size, max_len]
        
        # Create mask for real tokens (non-padding)
        mask = create_mask(lengths, max_len)
        
        # Count correct predictions only on real tokens
        correct_tokens += ((pred_tags == targets) & mask).sum().item()
        total_tokens += mask.sum().item()

accuracy = correct_tokens / total_tokens
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Correct tokens: {correct_tokens} / {total_tokens}")

# %%



