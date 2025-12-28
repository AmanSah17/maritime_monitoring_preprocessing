# TrAISformer Replication Guide for U.S. Maritime Data

## Overview
**Paper:** "TrAISformer -- A Transformer Network with Sparse Augmented Data Representation and Cross Entropy Loss for AIS-based Vessel Trajectory Prediction" (arXiv:2109.03958)

**Authors:** Duong Nguyen, Ronan Fablet

**Key Achievement:** Predicts vessel trajectories 10+ hours ahead with <10 nautical mile error

---

## Part 1: Understanding the TrAISformer Algorithm

### 1.1 Core Concept
TrAISformer is a **modified transformer network** that predicts vessel trajectories by:
- Converting AIS data into a **discrete, high-dimensional representation**
- Using a **cross-entropy loss function** to handle heterogeneous and multimodal vessel motion
- Extracting **long-term temporal patterns** from historical AIS observations

### 1.2 Architecture Overview

#### Input Features (4D Space)
The model operates on **4 key AIS parameters** per timestep:
1. **Latitude (LAT)** - vessel position (north-south)
2. **Longitude (LON)** - vessel position (east-west)
3. **SOG** - Speed Over Ground (in knots)
4. **COG** - Course Over Ground (direction in degrees)

#### Discretization/Tokenization Step (HIGH-DIMENSIONAL REPRESENTATION)
Instead of predicting continuous values, TrAISformer **quantizes** each feature:

```
LAT_size = 250 bins    → n_lat_embd = 256-dimensional embedding
LON_size = 270 bins    → n_lon_embd = 256-dimensional embedding
SOG_size = 30 bins     → n_sog_embd = 128-dimensional embedding
COG_size = 72 bins     → n_cog_embd = 128-dimensional embedding

Total embedding dimension: 256 + 256 + 128 + 128 = 768
```

**Why discretize?**
- Converts regression → classification problem (easier for transformers)
- Captures multimodal distribution of vessel movements
- Enables cross-entropy loss (more appropriate than MSE for discrete categories)

#### Embedding Projection
```
[LAT_token, LON_token, SOG_token, COG_token] 
    ↓ (lookup in embedding tables)
[LAT_embedding (256-D), LON_embedding (256-D), SOG_embedding (128-D), COG_embedding (128-D)]
    ↓ (concatenate)
Single token representation (768-D)
```

### 1.3 Transformer Architecture

#### Components:
- **Positional Embeddings:** Learn absolute position in sequence (max_seqlen=120)
- **Causal Self-Attention:** Standard multi-head attention with causal masking
  - 8 attention heads
  - 8 transformer blocks
- **Feed-Forward Networks:** 2-layer MLPs (768 → 3072 → 768)
- **Layer Normalization & Dropout:** For regularization

#### Forward Pass:
```
Input sequence (batch_size, seq_len, 4)
    ↓ (discretize to token indices)
Token indices (batch_size, seq_len, 4)
    ↓ (embed each token)
Embeddings (batch_size, seq_len, 768)
    ↓ (add positional embeddings)
Embedded sequence (batch_size, seq_len, 768)
    ↓ (8 transformer blocks with causal attention)
Contextual representations (batch_size, seq_len, 768)
    ↓ (classification head)
Logits (batch_size, seq_len, full_size) where full_size = 250+270+30+72 = 622
    ↓ (split and softmax)
[lat_probs, lon_probs, sog_probs, cog_probs]
    ↓ (sample or argmax)
Next position predictions
```

### 1.4 Training Configuration

#### Data Normalization
Input features are **normalized to [0, 1)** range:
- **Latitude:** Min=55.5°, Max=58.0° (Denmark/Baltic region)
- **Longitude:** Min=10.3°, Max=13.0° (Denmark/Baltic region)
- **SOG:** Max=30 knots
- **COG:** 0-360° maps to 0-1 range

#### Loss Function: Cross-Entropy
```
L = CrossEntropyLoss(lat_logits, lat_targets) +
    CrossEntropyLoss(lon_logits, lon_targets) +
    CrossEntropyLoss(sog_logits, sog_targets) +
    CrossEntropyLoss(cog_logits, cog_targets)
```

**Why CE instead of MSE?**
- Handles multimodal distributions (vessels can turn left OR right)
- Each feature treated as classification task
- More robust to outliers

#### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Max Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 6e-4 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Weight Decay | 0.1 |
| LR Schedule | Linear warmup → cosine decay |
| Warmup Tokens | 512×20 |
| Sequence Length (train) | 36-120 timesteps |
| Initial Seq Length | 18 timesteps |

#### Data Filtering
- **Minimum SOG threshold:** 0.05 knots (filter stationary vessels)
- **Minimum sequence length:** 36 timesteps
- **Data quality:** Remove trajectories with NaN values

### 1.5 Sampling/Inference Strategy

#### Standard Sampling Modes

**"pos" mode (position prediction):**
- Predicts next position directly from logits
- Uses argmax or sampling from softmax distributions

**"pos_vicinity" mode (constrained sampling):**
- Only considers positions within a "vicinity" radius (e.g., r_vicinity=40)
- Prevents unrealistic jumps in position
- Better for longer-horizon predictions

**Top-K Filtering:**
- Optional: keep only top-k most likely positions
- Reduces computational cost during inference

#### Autoregressive Prediction
```
Input: [t0, t1, ..., t_n]
↓ predict next
Pred: [t0, t1, ..., t_n, t_{n+1}]
↓ feed back + predict again
Pred: [t0, t1, ..., t_n, t_{n+1}, t_{n+2}]
... (repeat for desired forecast horizon)
```

### 1.6 Performance Metrics

The paper evaluates on:
- **Spatial Error:** Haversine distance (kilometers converted to nautical miles)
  - Formula: $d = 2R \sin^{-1}\sqrt{\sin^2(\Δ\phi/2) + \cos(\phi_1)\cos(\phi_2)\sin^2(\Δ\lambda/2)}$
  - Where R = 6371 km
- **Prediction Horizons:** 1hr, 3hr, 6hr, 10hr
- **Target Accuracy:** <10 nautical miles at 10-hour horizon

---

## Part 2: Data Preprocessing Steps (The Paper's Approach)

### 2.1 Raw AIS Data Requirements

The original paper uses **Danish Maritime Authority (DMA) data**:
```
Fields needed:
- MMSI: Vessel identifier
- Latitude: Navigation position
- Longitude: Navigation position
- SOG: Speed Over Ground (knots)
- COG: Course Over Ground (degrees 0-360)
- Timestamp: Unix or ISO format (for sorting)
- Any other optional: vessel type, draught, destination, etc.
```

### 2.2 Data Preprocessing Pipeline

#### Step 1: Load and Sort
```python
df = load_ais_data()
df = df.sort_values(['MMSI', 'Timestamp'])
```

#### Step 2: Trajectory Segmentation by Vessel
```python
for mmsi, vessel_traj in df.groupby('MMSI'):
    # Each vessel's trajectory is separate
    # Ensures temporal continuity within vessel
```

#### Step 3: Remove Stationary Vessels
```python
min_sog_threshold = 0.05  # knots
moving_idx = np.where(traj[:, SOG] > 0.05)[0][0]
traj = traj[moving_idx:, :]
```

#### Step 4: Quality Checks
```python
# Remove trajectories with:
- NaN values in any feature
- Gaps that exceed interpolation limits
- Sequences shorter than min_seqlen (36 timesteps)
```

#### Step 5: Normalization (Geographic Region Dependent)
```python
# For Denmark/Baltic region:
lat_normalized = (lat - 55.5) / (58.0 - 55.5)
lon_normalized = (lon - 10.3) / (13.0 - 10.3)
sog_normalized = sog / 30.0
cog_normalized = cog / 360.0

# Clip to [0, 1) range
x_normalized = np.clip(x_normalized, 0, 0.9999)
```

#### Step 6: Create Discrete Tokens
```python
# Uniform quantization
lat_token = int(lat_normalized * 250)        # 0-249
lon_token = int(lon_normalized * 270)        # 0-269
sog_token = int(sog_normalized * 30)         # 0-29
cog_token = int(cog_normalized * 72)         # 0-71
```

### 2.3 Dataset Structure (Pickle Format)

The paper stores preprocessed data as Python pickle files containing **lists of dictionaries**:

```python
# Format from the code:
[
    {
        "mmsi": 123456789,
        "traj": numpy array of shape (N, 5)
                Columns: [LAT_norm, LON_norm, SOG_norm, COG_norm, TIMESTAMP]
                All normalized to [0, 1)
    },
    {
        "mmsi": 987654321,
        "traj": ...
    },
    ...
]
```

### 2.4 Train/Validation/Test Split

Three separate pickle files created:
- **train.pkl:** ~80% of vessel trajectories
- **valid.pkl:** ~10% of vessel trajectories
- **test.pkl:** ~10% of vessel trajectories

Split should be **by MMSI** (vessel level) to avoid temporal leakage!

---

## Part 3: Implementation Plan for U.S. Maritime Data

### 3.1 Data Preparation Steps

#### Step 1: Determine Geographic Bounds for U.S. Waters
You need to identify the **min/max latitude and longitude** for your U.S. region:

**Example regions:**
- U.S. East Coast: lat 25-45°N, lon -80° to -70°W
- U.S. West Coast: lat 32-49°N, lon -130° to -117°W
- Gulf of Mexico: lat 25-30°N, lon -97° to -83°W

```python
# Analyze your processed_data/*.csv files
import pandas as pd

df = pd.read_csv('processed_data/AIS_2020_01_05.csv')

lat_min = df['LAT'].min()
lat_max = df['LAT'].max()
lon_min = df['LON'].min()
lon_max = df['LON'].max()
sog_min = df['SOG'].min()
sog_max = df['SOG'].max()

print(f"LAT range: {lat_min} to {lat_max}")
print(f"LON range: {lon_min} to {lon_max}")
print(f"SOG range: {sog_min} to {sog_max}")
```

#### Step 2: Create Data Configuration
```python
# In config_trAISformer.py:
if dataset_name == "us_maritime":
    lat_size = 250  # or calculate: int((lat_max - lat_min) * 10)
    lon_size = 300  # typically larger for US region
    sog_size = 30   # knots
    cog_size = 72   # 0-360 degrees
    
    lat_min = -lat_range_min  # e.g., 25.0
    lat_max = -lat_range_max  # e.g., 45.0
    lon_min = lon_range_min   # e.g., -130.0
    lon_max = lon_range_max   # e.g., -70.0
```

#### Step 3: Convert Your Interpolated CSV Data to TrAISformer Format

Create a preprocessing script to:

```python
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Configuration
LAT_MIN, LAT_MAX = 25.0, 45.0
LON_MIN, LON_MAX = -130.0, -70.0
SOG_MAX = 30.0

# Load and concatenate all CSV files
df_list = []
for csv_file in glob.glob('processed_data/AIS_*.csv'):
    df_list.append(pd.read_csv(csv_file))

df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)

# Process by vessel
trajectories = []
for mmsi, group in tqdm(df.groupby('MMSI')):
    # Filter moving vessels
    moving_indices = group['SOG'] > 0.05
    group = group[moving_indices].reset_index(drop=True)
    
    if len(group) < 36:  # minimum sequence length
        continue
    
    # Normalize features
    traj_array = np.zeros((len(group), 5))
    traj_array[:, 0] = (group['LAT'].values - LAT_MIN) / (LAT_MAX - LAT_MIN)
    traj_array[:, 1] = (group['LON'].values - LON_MIN) / (LON_MAX - LON_MIN)
    traj_array[:, 2] = group['SOG'].values / SOG_MAX
    traj_array[:, 3] = group['COG'].values / 360.0
    traj_array[:, 4] = group['BaseDateTime'].astype(np.int64).values // 10**9
    
    # Clip to [0, 0.9999]
    traj_array[:, :4] = np.clip(traj_array[:, :4], 0, 0.9999)
    
    trajectories.append({
        "mmsi": int(mmsi),
        "traj": traj_array
    })

# Train/val/test split (80/10/10 by vessel count)
n_vessels = len(trajectories)
n_train = int(0.8 * n_vessels)
n_val = int(0.1 * n_vessels)

train_data = trajectories[:n_train]
val_data = trajectories[n_train:n_train+n_val]
test_data = trajectories[n_train+n_val:]

# Save as pickle
with open('data/us_maritime/us_maritime_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('data/us_maritime/us_maritime_valid.pkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('data/us_maritime/us_maritime_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)
```

### 3.2 Model Training Configuration

#### Adjust hyperparameters for U.S. region

```python
# config_trAISformer.py modifications

class Config():
    # Data dimensions (adjust based on your geographic bounds)
    if dataset_name == "us_maritime":
        lat_size = 250
        lon_size = 300  # US region is wider
        sog_size = 30
        cog_size = 72
        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        
        # IMPORTANT: Update to your U.S. bounds
        lat_min = 25.0      # Southernmost point
        lat_max = 45.0      # Northernmost point
        lon_min = -130.0    # Westernmost point
        lon_max = -70.0     # Easternmost point
        
    # Keep other parameters same
    max_epochs = 50
    batch_size = 32
    learning_rate = 6e-4
    max_seqlen = 120
    min_seqlen = 36
```

#### Expected Training Time
- **GPU:** ~2-4 hours on V100/A100 per epoch
- **Total:** ~100-200 hours for 50 epochs
- **Dataset size:** 500K-1M vessel trajectories needed

### 3.3 Validation & Testing

#### Metrics to Compute

```python
# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in kilometers"""
    R = 6371  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

# Convert to nautical miles
km_to_nm = 0.539957

# Compute errors at different horizons
def evaluate_predictions(model, test_loader, horizons=[1, 3, 6, 10]):
    """
    horizons: list of prediction steps (hours)
    Assumes 6-minute sampling (10 points per hour)
    """
    errors_by_horizon = {h: [] for h in horizons}
    
    for batch in test_loader:
        seq, mask, seqlen, mmsi, time_start = batch
        
        # Feed first 18 timesteps
        context = seq[:, :18, :]
        
        # Predict next points
        for horizon in horizons:
            n_steps = int(horizon * 10)  # 10 predictions per hour
            
            # Generate predictions
            predictions = sample(model, context, steps=n_steps, 
                               sample_mode="pos_vicinity", r_vicinity=40)
            
            # Get actual ground truth
            actual = seq[:, 18+n_steps, :2]  # lat, lon
            
            # Convert from normalized to actual coordinates
            pred_lat = (predictions[:, -1, 0] * (LAT_MAX - LAT_MIN)) + LAT_MIN
            pred_lon = (predictions[:, -1, 1] * (LON_MAX - LON_MIN)) + LON_MIN
            
            actual_lat = (actual[:, 0] * (LAT_MAX - LAT_MIN)) + LAT_MIN
            actual_lon = (actual[:, 1] * (LON_MAX - LON_MIN)) + LON_MIN
            
            # Calculate error
            distance_km = haversine_distance(actual_lat, actual_lon, 
                                           pred_lat, pred_lon)
            distance_nm = distance_km * km_to_nm
            
            errors_by_horizon[h].append(distance_nm.mean())
    
    return errors_by_horizon
```

### 3.4 File Structure for U.S. Implementation

```
TRAIS_Former_/
├── CEE_TrAISformer/
│   ├── config_trAISformer.py          # ← Update with US bounds
│   ├── models.py                       # (no changes needed)
│   ├── datasets.py                     # (no changes needed)
│   ├── trainers.py                     # (no changes needed)
│   ├── trAISformer.py                 # (no changes needed)
│   ├── utils.py                        # (no changes needed)
│   └── data/
│       └── us_maritime/               # ← Create this
│           ├── us_maritime_train.pkl  # ← Generate from your CSV
│           ├── us_maritime_valid.pkl
│           └── us_maritime_test.pkl
│
├── prep_us_maritime.py                # ← Create preprocessing script
└── us_maritime_config.py              # ← Create custom config
```

---

## Part 4: Quick Start Implementation Checklist

### Phase 1: Data Preparation (Week 1)
- [ ] Analyze your processed_data/ to determine geographic bounds
- [ ] Create preprocessing script to convert CSV → pickle format
- [ ] Normalize features to [0, 1) range
- [ ] Create train/val/test split (80/10/10 by vessel)
- [ ] Verify pickle files contain correct structure

### Phase 2: Configuration (Day 1)
- [ ] Create modified config_trAISformer.py for U.S. data
- [ ] Update lat/lon min/max values
- [ ] Adjust lat_size/lon_size based on geographic bounds
- [ ] Set appropriate paths to pickle files

### Phase 3: Training (Week 2-4)
- [ ] Run trAISformer.py to train model
- [ ] Monitor loss curves for convergence
- [ ] Save best model checkpoint
- [ ] Expected time: 100-200 GPU hours

### Phase 4: Evaluation (Week 4)
- [ ] Implement haversine distance calculation
- [ ] Evaluate at 1h, 3h, 6h, 10h horizons
- [ ] Compare to baseline models (LSTM, simple extrapolation)
- [ ] Generate error distribution plots

---

## Part 5: Key Differences from Original Paper

When adapting to U.S. maritime data:

1. **Geographic Scale:** U.S. waters are much larger (both lat/lon extent)
   - May need larger bin sizes (lon_size=300+ vs 270)
   
2. **Vessel Diversity:** More heterogeneous traffic patterns
   - Consider multi-task learning if vessel types vary significantly
   
3. **Data Resolution:** Verify sampling rate
   - Original: ~6 minute intervals (10 points/hour)
   - Your data: Check and adjust accordingly
   
4. **Seasonal Variation:** U.S. waters have more seasonal variation
   - May need separate models or stratified training
   
5. **Computational Cost:** Larger region → more epochs may be needed
   - Budget 200+ GPU hours

---

## References & Resources

1. **Original Paper:** arXiv:2109.03958
2. **Code Base:** https://github.com/CIA-Oceanix/trAISformer
3. **Related Work:** GeoTrackNet (https://github.com/CIA-Oceanix/GeoTrackNet)
4. **Data Source:** Danish Maritime Authority (DMA)
5. **Transformer Reference:** minGPT (https://github.com/karpathy/minGPT)

---

## Questions & Troubleshooting

**Q: Why discretize instead of regression?**
A: Transformers work better with classification tasks. Cross-entropy loss handles multimodal distributions better than MSE for vessel trajectories that can take multiple valid paths.

**Q: What if my geographic bounds are different?**
A: Adjust lat_min, lat_max, lon_min, lon_max in config. The bin sizes (lat_size, lon_size) should scale proportionally.

**Q: How many trajectories do I need?**
A: Paper uses ~100K trajectories from DMA. For good U.S. model, aim for 500K+ distinct vessel trajectories.

**Q: Can I use different input features?**
A: The architecture is designed for exactly 4 features (lat, lon, sog, cog). Adding features requires architectural changes.

**Q: How do I handle vessels that stop?**
A: Filter with SOG threshold (0.05 knots). Stationary vessels aren't useful for trajectory prediction.
