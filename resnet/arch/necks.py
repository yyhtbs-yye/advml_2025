import torch
import torch.nn as nn

# Feature Pyramid Network (FPN) Neck
class FPNNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels, final_only=True):
        super(FPNNeck, self).__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.num_levels = len(in_channels_list)
        self.final_only = final_only
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, inputs):
        
        # Apply lateral convolutions to all features - from deepest to shallowest
        laterals = []
        for i, feature in enumerate(reversed(inputs)):  # Process from deepest (last) to shallowest (first)
            laterals.append(self.lateral_convs[self.num_levels - 1 - i](feature))
        
        # Top-down pathway
        # Start with the deepest feature and work upward
        fpn_features = [laterals[0]]  # Start with deepest level - no fusion needed
        
        # Process remaining levels by upsampling and adding
        for i in range(1, self.num_levels):
            # Upsample previous level to current level's size
            upsampled = nn.Upsample(size=laterals[i].shape[2:], mode='nearest')(fpn_features[-1])
            # Add lateral connection
            fused = laterals[i] + upsampled
            fpn_features.append(fused)
        
        # Apply post-fusion 3x3 convs to get final outputs
        outputs = []
        # Process from shallow to deep (reverse the order of fpn_features)
        for i, feature in enumerate(reversed(fpn_features)):
            # Apply the corresponding conv
            processed_feature = self.fpn_convs[i](feature)
            outputs.append(processed_feature)
        
        
        return outputs[0] if self.final_only else outputs
