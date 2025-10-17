import torch
import torch.nn as nn


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
