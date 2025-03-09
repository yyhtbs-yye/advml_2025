import torch
import torch.nn as nn
from arch.resblocks import BasicBlock, Bottleneck
from arch.backbones import ResNetBackbone
from arch.necks import FPNNeck
from arch.heads import ClassificationHead
from arch.plug_n_play import *

# Complete ResNet with modular structure 
class FPNResNet(nn.Module):
    def __init__(self, block_class, num_classes, layers=[3, 4, 6, 3], fpn_out_channels=256, 
                 backbone_channels=[64, 128, 256, 512], hidden_dim=512, 
                 dropout_prob=0.5, use_dropout=True, pnp_class=None):
        super(FPNResNet, self).__init__()
        
        # Backbone network (returns features at all levels)
        self.backbone = ResNetBackbone(
            block_class=block_class,            # Basic block or bottleneck block.
            layers=layers,                      # Number of layers in each stage.        
            pnp_class=pnp_class,                # Post processing after each layer. 
            channels_list=backbone_channels,    # Number of channels in each layer.
        )
        
        # FPN neck
        self.fpn = FPNNeck(
            in_channels_list=backbone_channels, # Number of channels in each layer.
            out_channels=fpn_out_channels,
            final_only=True
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            in_channels=fpn_out_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_dropout=use_dropout
        )
        
    def forward(self, x):
        # Get features from backbone (list of feature maps)
        backbone_features = self.backbone(x)
        
        # Process features with FPN
        fpn_features = self.fpn(backbone_features)
                        
        # Apply classification head
        cls_output = self.classifier(fpn_features)
        
        return cls_output
    
def build_model(config): # config is a dictionary containing model parameters
    # Extract parameters from config
    backbone_depth = config.get('backbone_depth', 50)
    num_classes = config.get('num_classes', 1000)
    fpn_out_channels = config.get('fpn_out_channels', 256)
    hidden_dim = config.get('hidden_dim', 512)
    dropout_prob = config.get('dropout_prob', 0.5)
    use_dropout = config.get('use_dropout', True)
    # import according to string name, the file name is plug_n_play.py
    if config.get('pnp_class') is not None:
        pnp_class = getattr(__import__('arch').plug_n_play, config.get('pnp_class'))
    else:
        pnp_class = None

    # Determine backbone layers based on depth
    if backbone_depth == 18:
        layers = [2, 2, 2, 2]
        backbone_channels = [64, 128, 256, 512]
        block_class = BasicBlock
    elif backbone_depth == 34:
        layers = [3, 4, 6, 3]
        backbone_channels = [64, 128, 256, 512]
        block_class = BasicBlock
    elif backbone_depth == 50:
        layers = [3, 4, 6, 3]
        backbone_channels = [256, 512, 1024, 2048]
        block_class = Bottleneck
    elif backbone_depth == 101:
        layers = [3, 4, 23, 3]
        backbone_channels = [256, 512, 1024, 2048]
        block_class = Bottleneck
    elif backbone_depth == 152:
        layers = [3, 8, 36, 3]
        backbone_channels = [256, 512, 1024, 2048]
        block_class = Bottleneck
    else:
        raise ValueError(f"Unsupported backbone depth: {backbone_depth}")
    
    # Create model
    model = FPNResNet(
        num_classes=num_classes,
        block_class=block_class,
        layers=layers,
        fpn_out_channels=fpn_out_channels,
        backbone_channels=backbone_channels,
        hidden_dim=hidden_dim,
        dropout_prob=dropout_prob,
        use_dropout=use_dropout,
        pnp_class=pnp_class)
    
    return model

# Example usage
if __name__ == "__main__":
    config = {
        'backbone_depth': 101,
        'num_classes': 10,
        'fpn_out_channels': 256,
        'hidden_dim': 512,
        'dropout_prob': 0.5,
        'use_dropout': 'SEModule',
        'pnp_class': 'CBAMBlock',
    }

    resnet = build_model(config)

    # test code 
    # Generate a random input tensor
    input_tensor = torch.randn(16, 3, 224, 224)
    # Forward pass through the model
    output = resnet(input_tensor)
    # Print the output shape
    print(output.shape)  # Output shape should be (1, 10)