import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=512, dropout_prob=0.5, use_dropout=True):
        super(ClassificationHead, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x.shape=(batch_size, in_channels, H, W)
        x = self.gap(x)  # Global Average Pooling: 
        # x.shape=(batch_size, in_channels, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, in_channels)
        # x.shape=(batch_size, in_channels)
        x = self.mlp(x)  # Apply MLP
        return x