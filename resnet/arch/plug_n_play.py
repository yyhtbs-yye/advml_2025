import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Module
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ECAModule(nn.Module):
    """
    ECA Module:
    - Avoids explicit reduction (like in SE).
    - Uses a fast 1D convolution on the channel descriptor.
    """
    def __init__(self, channels, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # [B, C, 1, 1]
        # Convert to [B, 1, C] for 1D conv
        y = y.view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y)
        # Reshape to [B, C, 1, 1]
        y = y.view(b, c, 1, 1)
        return x * y
    
import torch
import torch.nn as nn

class NonLocalBlock(nn.Module):
    """
    Implements the Non-local operation:
      y_i = (1/C(x)) * \sum_j f(x_i, x_j) g(x_j)
    Where f is typically a dot-product or embedded Gaussian function.
    """
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        
        # initialization trick
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, inter_C, H*W]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [B, inter_C, H*W]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, inter_C, H*W]
        
        # [B, inter_C, H*W] -> [B, H*W, inter_C]
        theta_x = theta_x.permute(0, 2, 1)
        # f = theta_x * phi_x
        f = torch.matmul(theta_x, phi_x)  # [B, H*W, H*W]

        # normalize by size for stability
        f_div_C = F.softmax(f, dim=-1)

        # y = f * g
        y = torch.matmul(f_div_C, g_x.permute(0, 2, 1))  # [B, H*W, inter_C]
        y = y.permute(0, 2, 1).contiguous()  # [B, inter_C, H*W]
        y = y.view(batch_size, self.inter_channels, H, W)
        y = self.W(y)
        return x + y
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial Attention
        sa = self.spatial_attention(x)
        x = x * sa
        return x
