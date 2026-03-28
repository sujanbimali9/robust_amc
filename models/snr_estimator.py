import torch
import torch.nn as nn
import torch.nn.functional as F


class SNREstimator(nn.Module):
    def __init__(self, input_channels=2, hidden_dims=[128, 64], output_dim=3):
        """
        SNR Estimator network that predicts SNR bin probabilities.
        Uses statistical features alongside learned CNN features for robustness.
        
        Args:
            input_channels: Number of input channels (2 for I/Q)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of SNR bins (3 for low/mid/high)
        """
        super(SNREstimator, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Statistical features: 6 hand-crafted features
        # (mean_power, peak_to_avg, kurtosis_I, kurtosis_Q, variance_I, variance_Q)
        stat_dim = 6
        cnn_dim = 128

        # Fully connected layers (CNN features + statistical features)
        fc_layers = []
        prev_dim = cnn_dim + stat_dim
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

    def _compute_stats(self, x):
        """Compute hand-crafted statistical features from I/Q signal"""
        # x: (batch, 2, L)
        I = x[:, 0, :]  # (batch, L)
        Q = x[:, 1, :]  # (batch, L)
        
        power = (I ** 2 + Q ** 2)  # (batch, L)
        mean_power = power.mean(dim=1, keepdim=True)
        peak_power = power.max(dim=1, keepdim=True)[0]
        
        # Peak-to-average power ratio (PAPR) — indicator of SNR
        papr = peak_power / (mean_power + 1e-8)
        
        # Kurtosis of I and Q channels
        I_centered = I - I.mean(dim=1, keepdim=True)
        Q_centered = Q - Q.mean(dim=1, keepdim=True)
        
        I_var = (I_centered ** 2).mean(dim=1, keepdim=True) + 1e-8
        Q_var = (Q_centered ** 2).mean(dim=1, keepdim=True) + 1e-8
        
        I_kurt = ((I_centered ** 4).mean(dim=1, keepdim=True)) / (I_var ** 2) - 3.0
        Q_kurt = ((Q_centered ** 4).mean(dim=1, keepdim=True)) / (Q_var ** 2) - 3.0
        
        stats = torch.cat([mean_power, papr, I_kurt, Q_kurt, I_var, Q_var], dim=1)
        return stats

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 2, sample_length)
        Returns:
            SNR bin logits (batch_size, num_bins) — NO softmax applied.
        """
        # Statistical features
        stats = self._compute_stats(x)
        
        # CNN features
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = self.pool1(feat)

        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = self.pool2(feat)

        feat = F.relu(self.bn3(self.conv3(feat)))
        feat = self.pool3(feat)

        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)

        # Concatenate CNN features with statistical features
        combined = torch.cat([feat, stats], dim=1)
        out = self.fc(combined)
        return out


class SNRRegressorEstimator(nn.Module):
    def __init__(self, input_channels=2, hidden_dims=[128, 64]):
        """
        SNR Estimator that directly regresses SNR value
        """
        super(SNRRegressorEstimator, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        fc_layers = []
        prev_dim = 128
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x