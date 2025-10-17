# src/models.py
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Simple Temporal Convolutional Network (TCN-like)
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, output_dim=5, num_channels=[64,64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for ch in num_channels:
            layers += [nn.Conv1d(in_ch, ch, kernel_size, padding=(kernel_size-1)//2),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(in_ch, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        # x: (B, T, F) -> Conv1d expects (B, F, T)
        x = x.permute(0,2,1)
        out = self.net(x)  # (B, C, T)
        out = out.mean(dim=2)  # global average over time
        return self.fc(out)

def model_factory(name, input_dim, **kwargs):
    if name.lower() == 'lstm':
        return LSTMModel(input_dim, **kwargs)
    elif name.lower() == 'tcn':
        return TemporalConvNet(input_dim, **kwargs)
    else:
        raise ValueError("Unknown model name")
