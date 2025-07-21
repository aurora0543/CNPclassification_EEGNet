# models.py

import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution.
    This implementation mimics Keras's SeparableConv2D layer. It first applies a
    depthwise convolution (a grouped convolution with groups=in_channels)
    followed by a pointwise convolution (a 1x1 convolution).
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    This block adaptively recalibrates channel-wise feature responses by
    explicitly modelling interdependencies between channels.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EEGNet(nn.Module):
    """
    EEGNet model adapted for multi-band frequency input.
    The input shape is expected to be (N, FreqBands, Chans, Samples).
    """
    def __init__(self, nb_classes, FreqBands=4, Chans=64, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, dropoutType='Dropout'):
        super().__init__()
        if dropoutType == 'SpatialDropout2D':
            self.dropout_layer = nn.Dropout2d
        else:
            self.dropout_layer = nn.Dropout

        # Block 1: Temporal and Spatial Filtering
        self.block1 = nn.Sequential(
            nn.Conv2d(FreqBands, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise Convolution learns spatial filters for each temporal filter
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.AvgPool2d((1, 4)),
            self.dropout_layer(dropoutRate)
        )
        self.depthwise_conv = self.block1[2]
        
        # Block 2: Depthwise Separable Convolution
        self.block2 = nn.Sequential(
            SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            self.dropout_layer(dropoutRate)
        )
        
        # Block 3: Channel-wise Attention
        self.attention = SEBlock(F2)
        
        # Dynamically calculate the flattened feature size for the classifier
        with torch.no_grad():
            dummy_input = torch.zeros(1, FreqBands, Chans, Samples)
            output_size = self.extract_features(dummy_input).shape[1]
            
        self.classifier = nn.Linear(output_size, nb_classes)

    def extract_features(self, x):
        """
        Public method to forward pass through the feature extraction layers.
        This can be used for pre-training or transfer learning.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.attention(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        """The main forward pass for classification."""
        features = self.extract_features(x)
        output = self.classifier(features)
        return output

    def apply_constraints(self):
        """Applies a max-norm constraint to the weights of the depthwise convolution layer."""
        self.depthwise_conv.weight.data = torch.renorm(
            self.depthwise_conv.weight.data, p=2, dim=0, maxnorm=1.0
        )