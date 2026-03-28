import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with optional channel projection and SE attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 use_se=True, dropout=0.0):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Shortcut projection if dimensions change
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class ExpertCNN(nn.Module):
    def __init__(self, input_channels=2, num_classes=8, filters=[64, 128, 256]):
        """
        Expert CNN classifier with residual blocks and SE attention.
        Optimized for 128-length I/Q signals.
        
        Args:
            input_channels: Number of input channels (2 for I/Q)
            num_classes: Number of modulation classes
            filters: List of filter sizes for conv stages
        """
        super(ExpertCNN, self).__init__()

        # Stem: initial feature extraction with multi-scale kernels
        self.stem_3 = nn.Conv1d(input_channels, filters[0] // 2, kernel_size=3, padding=1, bias=False)
        self.stem_7 = nn.Conv1d(input_channels, filters[0] // 2, kernel_size=7, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm1d(filters[0])
        
        # Stage 1: filters[0] channels, no downsampling
        self.stage1 = nn.Sequential(
            ResidualBlock(filters[0], filters[0], kernel_size=7, dropout=0.1),
            ResidualBlock(filters[0], filters[0], kernel_size=5, dropout=0.1),
        )

        # Stage 2: filters[0] -> filters[1], downsample x2
        self.stage2 = nn.Sequential(
            ResidualBlock(filters[0], filters[1], kernel_size=5, stride=2, dropout=0.1),
            ResidualBlock(filters[1], filters[1], kernel_size=3, dropout=0.1),
        )

        # Stage 3: filters[1] -> filters[2], downsample x2
        self.stage3 = nn.Sequential(
            ResidualBlock(filters[1], filters[2], kernel_size=3, stride=2, dropout=0.15),
            ResidualBlock(filters[2], filters[2], kernel_size=3, dropout=0.15),
        )

        # Global average + max pooling (concatenated for richer features)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        # Classifier head with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(filters[2] * 2, 256),  # *2 because avg+max pooling
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 2, sample_length)
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Multi-scale stem
        x3 = self.stem_3(x)
        x7 = self.stem_7(x)
        x = torch.cat([x3, x7], dim=1)
        x = F.relu(self.stem_bn(x))
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Combined pooling
        x_avg = self.gap(x).view(x.size(0), -1)
        x_max = self.gmp(x).view(x.size(0), -1)
        x = torch.cat([x_avg, x_max], dim=1)
        
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """Extract intermediate features (for gating network)"""
        x3 = self.stem_3(x)
        x7 = self.stem_7(x)
        x = torch.cat([x3, x7], dim=1)
        x = F.relu(self.stem_bn(x))
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x_avg = self.gap(x).view(x.size(0), -1)
        x_max = self.gmp(x).view(x.size(0), -1)
        return torch.cat([x_avg, x_max], dim=1)


class ResNetExpert(nn.Module):
    def __init__(self, input_channels=2, num_classes=8, base_filters=64):
        """
        Deeper ResNet-based expert classifier
        """
        super(ResNetExpert, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(base_filters, base_filters),
            ResidualBlock(base_filters, base_filters),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(base_filters, base_filters * 2, stride=2),
            ResidualBlock(base_filters * 2, base_filters * 2),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(base_filters * 2, base_filters * 4, stride=2),
            ResidualBlock(base_filters * 4, base_filters * 4),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_filters * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_avg = self.gap(x).view(x.size(0), -1)
        x_max = self.gmp(x).view(x.size(0), -1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.fc(x)
        return x