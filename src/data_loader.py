# src/data_loader.py
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class VesselSequenceDataset(Dataset):
    def __init__(self, df, input_features, target_features, seq_len=10, pred_len=1, min_points=12):
        self.samples = []
        self.input_features = input_features
        self.target_features = target_features
        for mmsi, grp in tqdm(df.groupby('MMSI'), desc="group_by_mmsi"):
            arr_in = grp[input_features].values
            arr_out = grp[target_features].values
            n = len(arr_in)
            if n < seq_len + pred_len: 
                continue
            for i in range(n - seq_len - pred_len + 1):
                x = arr_in[i:i+seq_len]
                y = arr_out[i+seq_len:i+seq_len+pred_len]
                self.samples.append((int(mmsi), x.astype('float32'), y.astype('float32').squeeze(0)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        mmsi, x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y), mmsi
