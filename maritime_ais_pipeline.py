import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import folium
from shapely.geometry import MultiPoint
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION
# ==============================
PKL_PATH = "processed_data/AIS_2020_01_03.pkl"
OUTPUT_DIR = "outputs_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# DATA LOADING
# ==============================
print("📦 Loading data...")
df = pd.read_pickle(PKL_PATH)

# ==============================
# DATA HEALTH CHECK
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("🔍 Analyzing data health (memory-safe mode)...")

# Only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
health_summary = []

# Iterate column-wise to avoid large memory allocations
for col in tqdm(numeric_cols, desc="Scanning numeric columns"):
    col_data = df[col]
    nan_count = col_data.isna().sum()
    inf_count = np.isinf(col_data.values).sum()
    mean_val = col_data.mean(skipna=True)
    std_val = col_data.std(skipna=True)
    min_val = col_data.min(skipna=True)
    max_val = col_data.max(skipna=True)
    health_summary.append([col, nan_count, inf_count, mean_val, std_val, min_val, max_val])

health_df = pd.DataFrame(
    health_summary,
    columns=["Feature", "NaN", "Inf", "Mean", "Std", "Min", "Max"]
).sort_values(by=["NaN", "Inf"], ascending=False)

print("\n📊 Top 15 Unstable Features:")
print(health_df.head(15).to_string(index=False))

# Optional: visualize a small subset (since huge dataset)
subset = df[numeric_cols[:20]].head(10000)  # first 20 cols, 10k rows
sns.heatmap(
    subset.isna() | np.isinf(subset),
    cmap="Reds",
    cbar=False
)
plt.title("🔥 Data Health Heatmap — NaN / Inf (first 20 features)")
plt.xlabel("Features")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()


# ==============================
# CLEAN DATA
# ==============================
print("🧹 Cleaning data...")
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e3, 1e3)

# ==============================
# FEATURE SCALING
# ==============================
print("⚙️ Scaling features...")
input_features = ["LAT", "LON", "SOG", "COG", "Heading", "v_x", "v_y", "turn_rate", "accel_knots_per_hr", "dayofweek", "month", "Δt_hours", "ΔCOG", "ΔSOG", "COG_rad"]
target_features = ["LAT", "LON", "SOG", "COG", "Heading"]

scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()
df[input_features] = scaler_in.fit_transform(df[input_features])
df[target_features] = scaler_out.fit_transform(df[target_features])

# ==============================
# KMEANS CLUSTERING
# ==============================
print("🧭 Performing KMeans clustering...")
coords = df[["LAT", "LON"]]
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(coords)

# ==============================
# MAP VISUALIZATION
# ==============================
print("🗺️ Generating Folium map...")
center = [df["LAT"].mean(), df["LON"].mean()]
m = folium.Map(location=center, zoom_start=5)
for cid, grp in df.groupby("cluster"):
    pts = MultiPoint(grp[["LON", "LAT"]].values)
    hull = pts.convex_hull
    if hull.geom_type == 'Polygon':
        folium.Polygon(locations=[(y, x) for x, y in hull.exterior.coords], color='blue', fill=True, opacity=0.3).add_to(m)
    folium.CircleMarker(location=center, radius=3, color='red', popup=f'Cluster {cid}').add_to(m)
m.save(f"{OUTPUT_DIR}/clusters_map.html")

# ==============================
# PCA
# ==============================
print("📊 Performing PCA analysis...")
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df[input_features])
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Explained Variance by PCA Components")
plt.xlabel("Component")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/pca_explained_variance.png")
plt.close()

loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(5)], index=input_features)
loadings.to_csv(f"{OUTPUT_DIR}/pca_loadings.csv")

# ==============================
# DATASET CLASS
# ==============================
class VesselSequenceDataset(Dataset):
    def __init__(self, df, seq_len=10, pred_len=1):
        self.samples = []
        self.input_features = input_features
        self.target_features = target_features
        for mmsi, group in tqdm(df.groupby("MMSI"), desc="Preparing sequences"):
            arr_in = group[self.input_features].values
            arr_out = group[self.target_features].values
            for i in range(len(arr_in) - seq_len - pred_len):
                X = arr_in[i:i+seq_len]
                y = arr_out[i+seq_len:i+seq_len+pred_len]
                self.samples.append((mmsi, X, y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        mmsi, X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), mmsi

# ==============================
# TRAIN-TEST SPLIT
# ==============================
print("✂️ Splitting train/test data...")
from sklearn.model_selection import train_test_split

full_dataset = VesselSequenceDataset(df, seq_len=10, pred_len=1)
idx = np.arange(len(full_dataset))
train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
train_ds = torch.utils.data.Subset(full_dataset, train_idx)
test_ds = torch.utils.data.Subset(full_dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# ==============================
# MODEL DEFINITION
# ==============================
class VesselLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=5, dropout=0.2):
        super(VesselLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ==============================
# TRAINING
# ==============================
print("🚀 Starting GPU training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VesselLSTM(len(input_features)).to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 25
train_losses = []

for epoch in tqdm(range(EPOCHS), desc="Training epochs"):
    model.train()
    epoch_loss = 0.0
    for X, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        X, y = X.to(device), y.squeeze(1).to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss/len(train_loader))
    torch.cuda.empty_cache()

plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/training_loss_curve.png")
plt.close()

print(f"✅ Training complete on device: {device}")
