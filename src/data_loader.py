import torch
from torch.utils.data import IterableDataset
import pandas as pd
from tqdm import tqdm

class LazyVesselSequenceDataset(IterableDataset):
    """
    Fully memory-efficient IterableDataset.
    Generates sequences lazily per vessel.
    """
    def __init__(self, df, input_features, target_features, seq_len=10, pred_len=1, device=None):
        self.df = df
        self.input_features = input_features
        self.target_features = target_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vessels = list(df['MMSI'].unique())

    def __iter__(self):
        # Shuffle vessels each epoch
        for mmsi in self.vessels:
            vessel_df = self.df[self.df['MMSI'] == mmsi].sort_values("BaseDateTime")
            features = torch.tensor(vessel_df[self.input_features].values, dtype=torch.float32)
            targets = torch.tensor(vessel_df[self.target_features].values, dtype=torch.float32)
            n_seq = len(features) - self.seq_len - self.pred_len + 1
            if n_seq <= 0:
                continue
            for i in range(n_seq):
                X = features[i:i+self.seq_len].to(self.device)
                y = targets[i+self.seq_len:i+self.seq_len+self.pred_len].to(self.device)
                if self.pred_len == 1:
                    y = y.squeeze(0)
                yield X, y
