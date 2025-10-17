import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AISDataset(Dataset):
    def __init__(self, pkl_path, seq_len=10, target_step=1, feature_cols=None, target_cols=None):
        self.df = pd.read_pickle(pkl_path)
        self.seq_len = seq_len
        self.target_step = target_step

        self.feature_cols = feature_cols 
        self.target_cols = target_cols or ["LAT", "LON", "SOG", "COG"]

        # Ensure sorted by MMSI and timestamp
        self.df = self.df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

        # Save group offsets for lazy access (without storing full group data)
        self.group_offsets = self.df.groupby("MMSI").indices
        self.mmsi_list = list(self.group_offsets.keys())

        # Precompute total sequence counts per vessel
        self.group_lengths = {
            mmsi: len(idxs) for mmsi, idxs in self.group_offsets.items()
        }

        self.total_sequences = sum(
            max(0, length - (seq_len + target_step))
            for length in self.group_lengths.values()
        )

        print(f"✅ Loaded {len(self.mmsi_list)} vessels, total sequences: {self.total_sequences}")

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Find which vessel this index belongs to
        for mmsi in self.mmsi_list:
            seq_count = max(0, self.group_lengths[mmsi] - (self.seq_len + self.target_step))
            if idx < seq_count:
                group_idx = self.group_offsets[mmsi]
                vessel_df = self.df.iloc[group_idx]
                start = idx
                end = start + self.seq_len
                target_idx = end + self.target_step

                X = vessel_df.iloc[start:end][self.feature_cols].values
                y = vessel_df.iloc[target_idx][self.target_cols].values
                return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
            idx -= seq_count

        raise IndexError("Index out of range")

