import torch
import torch.nn as nn

# Basic blocks for ResNet
class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, pnp_class=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.pnp = None if pnp_class is None else pnp_class(channels=out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.pnp is not None:
            out = self.pnp(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


# Bottleneck block in_channels -> out_channels//4 -> out_channels
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, pnp_class=None):
        super(Bottleneck, self).__init__()
        # The bottleneck design: in_channels -> out_channels/4 -> out_channels
        self.bottleneck_channels = out_channels // 4
        
        # First 1x1 conv reduces channels (in_channels -> bottleneck_channels)
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.bottleneck_channels)
        
        # 3x3 conv processes at reduced channels (bottleneck_channels -> bottleneck_channels)
        self.conv2 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        
        # Second 1x1 conv expands channels back (bottleneck_channels -> out_channels)
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.pnp = None if pnp_class is None else pnp_class(channels=out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.pnp is not None:
            out = self.pnp(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
