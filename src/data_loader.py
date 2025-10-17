import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

class VesselSequenceDataset(Dataset):
    """
    Memory-efficient AIS sequence dataset for time-series modeling.
    
    - Processes each MMSI group lazily (no full groupby materialization in memory)
    - Prepares sliding windows (seq_len → pred_len) sequences
    - Compatible with GPU tensors
    """

    def __init__(self, df, input_features, target_features, seq_len=10, pred_len=1, max_groups=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_features = input_features
        self.target_features = target_features

        # Convert to Pandas if needed (in case of Modin/cuDF)
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        # Sort just once — important for sequential consistency
        if 'BaseDateTime' in df.columns:
            df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)

        self.samples = []  # store (index_start, index_end, mmsi)
        self.data_store = {}  # optional in-memory caching for quick access

        # Unique MMSIs (lazy iteration avoids high memory use)
        unique_mmsis = df['MMSI'].unique()
        if max_groups:
            unique_mmsis = unique_mmsis[:max_groups]

        print(f"🛰️ Preparing sequences for {len(unique_mmsis)} vessels...")

        for mmsi in tqdm(unique_mmsis, desc="⛴️ Processing vessels", unit="vessel"):
            grp = df[df['MMSI'] == mmsi]
            if len(grp) < seq_len + pred_len:
                continue

            X = grp[input_features].to_numpy(dtype=np.float32)
            y = grp[target_features].to_numpy(dtype=np.float32)

            # Sliding window generation
            n_samples = len(X) - seq_len - pred_len + 1
            for i in range(n_samples):
                self.samples.append((mmsi, i))
            
            # Optionally keep numeric data (for speed) — but can skip to save memory
            self.data_store[mmsi] = (X, y)

            # clean up temporary references
            del grp, X, y
            gc.collect()

        print(f"✅ Total generated sequences: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mmsi, start_idx = self.samples[idx]
        X_all, y_all = self.data_store[mmsi]

        X_seq = X_all[start_idx:start_idx + self.seq_len]
        y_seq = y_all[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]

        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_seq = torch.tensor(y_seq, dtype=torch.float32)

        return X_seq, y_seq, mmsi
