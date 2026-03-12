import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM_1(nn.Module):

    def __init__(self, num_features: int = 6, num_classes: int = 6, hidden_dim: int = 64, lstm_layers: int = 1):
        """
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
            dropout=0.2
        )

        # Classification head over pooled temporal features.
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
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
    

class CNNLSTM_2(nn.Module):

    def __init__(self, num_features: int = 6, num_classes: int = 6, hidden_dim: int = 64, lstm_layers: int = 1):
        super().__init__()

        lstm_output_dim = hidden_dim * 2

        # Bidirectional LSTM for temporal context.
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )

        # 1D CNN feature extractor over LSTM outputs.
        self.conv1 = nn.Conv1d(in_channels=lstm_output_dim, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Classification head over pooled temporal features.
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, time, features)
        Output shape: (batch, num_classes)
        """
        # LSTM first: (batch, time, features) -> (batch, time, hidden_dim * 2).
        lstm_out, _ = self.lstm(x)

        # Conv1d expects channel-first: (batch, channels, time).
        x = lstm_out.permute(0, 2, 1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # Return to time-major feature layout for pooling.
        x = x.permute(0, 2, 1)

        # Combine mean and max temporal pooling for a richer sequence summary.
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        out = torch.cat([mean_pool, max_pool], dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out    
