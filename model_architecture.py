import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Single TCN residual block with dilated convolution."""

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, dilation: int = 1):
        """Build a dilated conv block with batch norm and residual connection.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            dilation: Dilation factor for the convolution.
        """
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.dropout = nn.Dropout(0.2)

        # Project input to output channels if needed for residual connection.
        self.residual_projection = None
        if input_channels != output_channels:
            self.residual_projection = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Input shape: (batch, channels, time)
        Output shape: (batch, output_channels, time)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        if self.residual_projection is not None:
            residual = self.residual_projection(residual)

        return out + residual


class TCN(nn.Module):
    """Temporal Convolutional Network for activity classification on IMU data."""

    def __init__(self, num_features: int = 6, num_classes: int = 6, num_layers: int = 4, num_channels: int = 64):
        """Build a stacked TCN with exponentially increasing dilation.

        Args:
            num_features: Number of input sensor channels per timestep.
            num_classes: Number of output activity classes.
            num_layers: Number of TCN blocks.
            num_channels: Base number of channels in each block.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # Input projection to num_channels.
        self.input_conv = nn.Conv1d(num_features, num_channels, kernel_size=1)

        # Stack TCN blocks with increasing dilation.
        for i in range(num_layers):
            dilation = 2**i
            in_ch = num_channels
            out_ch = num_channels * min(2, (i // 2) + 1)  # Gradually increase channels.

            self.layers.append(TCNBlock(in_ch, out_ch, kernel_size=3, dilation=dilation))
            num_channels = out_ch

        # Global average and max pooling + classification head.
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input shape: (batch, time, features)
        Output shape: (batch, num_classes)
        """
        # Conv1d expects channel-first: (batch, features, time).
        x = x.permute(0, 2, 1)

        # Input projection.
        x = self.input_conv(x)

        # Pass through TCN blocks.
        for layer in self.layers:
            x = layer(x)

        # Global temporal pooling.
        mean_pool = x.mean(dim=2)
        max_pool, _ = x.max(dim=2)
        out = torch.cat([mean_pool, max_pool], dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out


class CNNLSTM(nn.Module):
    """CNN + BiLSTM classifier for windowed IMU sequences."""

    def __init__(self, num_features: int = 6, num_classes: int = 6, hidden_dim: int = 64, lstm_layers: int = 1):
        """Build feature extractor + temporal model + classification head.

        Args:
            num_features: Number of input sensor channels per timestep.
            num_classes: Number of output activity classes.
            hidden_dim: Hidden size for the BiLSTM.
            lstm_layers: Number of stacked LSTM layers.
        """
        super().__init__()

        # 1D CNN feature extractor over time.
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Bidirectional LSTM for temporal context.
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )

        # Classification head over pooled temporal features.
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input shape: (batch, time, features)
        Output shape: (batch, num_classes)
        """
        # Conv1d expects channel-first: (batch, features, time).
        x = x.permute(0, 2, 1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # LSTM expects time-first per sample: (batch, time, channels).
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # Combine mean and max temporal pooling for a richer sequence summary.
        mean_pool = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        out = torch.cat([mean_pool, max_pool], dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out
