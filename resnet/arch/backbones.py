import torch
import torch.nn as nn

# ResNet Backbone
class ResNetBackbone(nn.Module):
    def __init__(self, block_class, layers=[3, 4, 6, 3], pnp_class=None, channels_list=[64, 128, 256, 512]):

        super(ResNetBackbone, self).__init__()
        self.block_class = block_class
        self.pnp_class = pnp_class
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, channels_list[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_list[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with channel sizes following standard ResNet architecture
        self.layer1 = self._make_layer(channels_list[0], channels_list[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels_list[0], channels_list[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels_list[1], channels_list[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels_list[2], channels_list[3], layers[3], stride=2)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    # channels_list[0], channels_list[0], layers[0], stride=1`
    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1):
        downsample = None
        
        # Need to downsample if stride > 1 or input/output channel dimensions don't match
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [self.block_class(in_channels, out_channels, stride, downsample, self.pnp_class)]
        
        for _ in range(1, n_blocks):
                layers.append(self.block_class(out_channels, out_channels, 1, None, self.pnp_class))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)      # 1/4 resolution
        c2 = self.layer2(c1)     # 1/8 resolution
        c3 = self.layer3(c2)     # 1/16 resolution
        c4 = self.layer4(c3)     # 1/32 resolution
        
        return [c1, c2, c3, c4]

