# Step-by-Step Implementation Guide for U.S. Maritime AIS Trajectory Prediction

## Phase 1: Analyze Your Data (Day 1)

### Step 1.1: Understand Your Data Format
First, examine your processed CSV files to understand the structure and bounds:

```python
# Script: analyze_us_maritime.py
import pandas as pd
import numpy as np
import glob

# Load one sample CSV to check structure
df_sample = pd.read_csv('processed_data/AIS_2020_01_05.csv')

print("CSV Columns:")
print(df_sample.columns.tolist())
print("\nFirst 5 rows:")
print(df_sample.head())
print("\nData types:")
print(df_sample.dtypes)
print("\nBasic statistics:")
print(df_sample.describe())

# Check for required fields
required_fields = ['MMSI', 'LAT', 'LON', 'SOG', 'COG', 'BaseDateTime']
for field in required_fields:
    if field not in df_sample.columns:
        print(f"⚠️  WARNING: Missing required field '{field}'")
        print(f"   Available fields: {df_sample.columns.tolist()}")
```

**What to look for:**
- Latitude range: e.g., 25-45°N
- Longitude range: e.g., -130° to -70°W
- SOG (Speed) range: should be 0-30+ knots
- COG (Course) range: should be 0-360°
- Total number of AIS records
- Number of unique vessels (MMSI values)

### Step 1.2: Determine Geographic Bounds

```python
# Script: find_bounds.py
import pandas as pd
import glob
import numpy as np

# Combine all CSV files
dfs = []
for csv_file in sorted(glob.glob('processed_data/AIS_*.csv')):
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Analyze bounds
print("="*60)
print("GEOGRAPHIC BOUNDS")
print("="*60)

lat_min = df_all['LAT'].min()
lat_max = df_all['LAT'].max()
lon_min = df_all['LON'].min()
lon_max = df_all['LON'].max()
sog_max = df_all['SOG'].max()

print(f"Latitude:  {lat_min:.2f}° to {lat_max:.2f}°")
print(f"Longitude: {lon_min:.2f}° to {lon_max:.2f}°")
print(f"SOG:       0 to {sog_max:.2f} knots")
print(f"\nLatitude range:  {lat_max - lat_min:.2f}°")
print(f"Longitude range: {lon_max - lon_min:.2f}°")

print("\n" + "="*60)
print("UPDATE prep_us_maritime.py WITH THESE VALUES:")
print("="*60)
print(f"""
LAT_MIN = {lat_min:.1f}
LAT_MAX = {lat_max:.1f}
LON_MIN = {lon_min:.1f}
LON_MAX = {lon_max:.1f}
SOG_MAX = {sog_max:.1f}
""")

# Estimate bin sizes
lat_range = lat_max - lat_min
lon_range = lon_max - lon_min
print(f"Recommended bin sizes:")
print(f"  lat_size = {int(lat_range * 10) + 10}")
print(f"  lon_size = {int(lon_range * 10) + 10}")
```

---

## Phase 2: Run Preprocessing (Days 2-3)

### Step 2.1: Update Preprocessing Configuration

Edit `prep_us_maritime.py` with your geographic bounds:

```python
# At top of prep_us_maritime.py, update:

# Geographic bounds - UPDATE WITH YOUR VALUES FROM PHASE 1
LAT_MIN = 25.0      # e.g., your minimum latitude
LAT_MAX = 45.0      # e.g., your maximum latitude
LON_MIN = -130.0    # e.g., your minimum longitude
LON_MAX = -70.0     # e.g., your maximum longitude
SOG_MAX = 30.0      # Maximum speed in knots
```

### Step 2.2: Run Preprocessing Script

**First run** (analysis only):
```bash
cd f:\PyTorch_GPU\maritime_monitoring_preprocessing
python prep_us_maritime.py
```

This will:
1. Scan all CSV files for bounds
2. Display the bounds analysis
3. Exit (asking you to configure bounds)

**After configuration:**
```bash
# Edit prep_us_maritime.py with bounds from analysis
# Then run again:
python prep_us_maritime.py
```

This will:
1. Load all CSV files (~5-10 minutes depending on size)
2. Preprocess trajectories (~10-20 minutes)
3. Create train/val/test split (80/10/10)
4. Save 3 pickle files (~100 MB-1 GB each)
5. Generate config template

**Expected output:**
```
[STEP 1] Analyzing data bounds...
[STEP 2] Loading AIS data...
✓ Loaded 500,000 total AIS records
✓ Found 5,000 unique vessels

[STEP 3] Preprocessing trajectories...
[STEP 4] Splitting data...
[STEP 5] Saving pickle files...
  ✓ us_maritime_train.pkl
    - 4,000 trajectories
    - 2,500,000 total timesteps
    - 625.0 avg length

✓ PREPROCESSING COMPLETE
```

### Step 2.3: Verify Pickle Files

```python
# Script: verify_pickle.py
import pickle
import numpy as np

# Check train pickle
with open('TRAIS_Former_/CEE_TrAISformer/data/us_maritime/us_maritime_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

print(f"Train set: {len(train_data)} trajectories")
print(f"First trajectory MMSI: {train_data[0]['mmsi']}")
print(f"First trajectory shape: {train_data[0]['traj'].shape}")
print(f"First trajectory (first 3 timesteps):")
print(train_data[0]['traj'][:3])

# Check value ranges
all_values = np.concatenate([traj['traj'][:, :4] for traj in train_data])
print(f"\nValue ranges (should be [0, 1)):")
print(f"  Min: {all_values.min():.4f}")
print(f"  Max: {all_values.max():.4f}")

# Statistics
total_timesteps = sum(len(traj['traj']) for traj in train_data)
avg_length = total_timesteps / len(train_data)
print(f"\nStatistics:")
print(f"  Total timesteps: {total_timesteps:,}")
print(f"  Average trajectory length: {avg_length:.1f}")
```

---

## Phase 3: Configure Model (Day 4)

### Step 3.1: Create U.S. Maritime Configuration

**Option A: Use Generated Config (Easier)**

The `prep_us_maritime.py` script generates a config template. Copy relevant sections:

```bash
# The script created a config at:
# TRAIS_Former_/CEE_TrAISformer/data/us_maritime/../../../config_us_maritime.py

# Copy to main config file (or create new)
copy TRAIS_Former_/CEE_TrAISformer/data/us_maritime/../../../config_us_maritime.py ^
     TRAIS_Former_/CEE_TrAISformer/config_us_maritime.py
```

**Option B: Manual Configuration (Fine Control)**

Create `TRAIS_Former_/CEE_TrAISformer/config_us_maritime.py`:

```python
# config_us_maritime.py
import os
import torch

class Config():
    """TrAISformer configuration for U.S. Maritime data"""
    
    # ==================== DEVICE & BASIC ====================
    retrain = True
    device = torch.device("cuda:0")  # or "cuda:1", "cuda:2" if multiple GPUs
    
    # ==================== TRAINING HYPERPARAMETERS ====================
    max_epochs = 50
    batch_size = 32              # Adjust if out of memory
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    
    # ==================== SEQUENCE PARAMETERS ====================
    init_seqlen = 18             # Initial conditioning sequence
    max_seqlen = 120             # Maximum sequence length
    min_seqlen = 36              # Minimum for training
    
    # ==================== DATASET SELECTION ====================
    dataset_name = "us_maritime"
    
    # ==================== U.S. MARITIME SPECIFIC ====================
    if dataset_name == "us_maritime":
        # GEOGRAPHIC BOUNDS (from your Phase 1 analysis)
        lat_min = 25.0           # ← UPDATE: Southern bound
        lat_max = 45.0           # ← UPDATE: Northern bound
        lon_min = -130.0         # ← UPDATE: Western bound
        lon_max = -70.0          # ← UPDATE: Eastern bound
        
        # QUANTIZATION BIN SIZES
        # Calculate as: (max - min) * ~10 bins per degree
        lat_range = lat_max - lat_min  # e.g., 20°
        lon_range = lon_max - lon_min  # e.g., 60°
        
        lat_size = int(lat_range * 12) + 5   # e.g., 245
        lon_size = int(lon_range * 4) + 20   # e.g., 260
        sog_size = 30                         # 0-30 knots
        cog_size = 72                         # 5° per bin
        
        # EMBEDDING DIMENSIONS
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
    
    # ==================== TRANSFORMER ARCHITECTURE ====================
    n_head = 8                   # Attention heads
    n_layer = 8                  # Transformer blocks
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    
    # Dropout
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # ==================== DATA PATHS ====================
    datadir = "./data/us_maritime/"
    trainset_name = "us_maritime_train.pkl"
    validset_name = "us_maritime_valid.pkl"
    testset_name = "us_maritime_test.pkl"
    
    # ==================== MODEL PARAMETERS ====================
    mode = "pos"                 # Position prediction
    sample_mode = "pos_vicinity" # Constrained sampling
    top_k = 10                   # Top-K filtering
    r_vicinity = 40              # Vicinity radius for constrained sampling
    
    # ==================== BLUR (OPTIONAL) ====================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    
    # ==================== OUTPUT ====================
    filename = (f"{dataset_name}"
                f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}"
                f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}"
                f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"
                f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"
                f"-head-{n_head}-{n_layer}"
                f"-bs-{batch_size}"
                f"-lr-{learning_rate}"
                f"-seqlen-{init_seqlen}-{max_seqlen}")
    
    savedir = "./results/" + filename + "/"
    ckpt_path = os.path.join(savedir, "model.pt")
    
    # ==================== WORKERS ====================
    num_workers = 4              # Reduce if out of memory
```

### Step 3.2: Update trAISformer.py if Needed

Check if config import is correct:

```python
# In TRAIS_Former_/CEE_TrAISformer/trAISformer.py, ensure:
from config_us_maritime import Config  # Instead of config_trAISformer
# OR just stick with: from config_trAISformer import Config
# and update config_trAISformer.py instead
```

**Easier approach:** Just update `config_trAISformer.py` directly with your U.S. values.

---

## Phase 4: Train Model (7-14 days on GPU)

### Step 4.1: Start Training

```bash
cd TRAIS_Former_/CEE_TrAISformer

# Run training
python trAISformer.py

# Or with output logging:
python trAISformer.py > training_log.txt 2>&1 &
```

**Expected training time:**
- **Per epoch:** 2-4 hours (on V100/A100)
- **Total (50 epochs):** 100-200 hours
- **If running on CPU:** 10-20x slower (not recommended)

### Step 4.2: Monitor Training

```bash
# Watch loss in real-time (Linux/Mac)
tail -f training_log.txt

# Or check tensorboard if enabled (in config)
tensorboard --logdir=./runs/
```

**Expected loss curve:**
```
Epoch 0:  Loss: 6.234  (high - random predictions)
Epoch 5:  Loss: 3.456  (steady decrease)
Epoch 10: Loss: 2.156
Epoch 20: Loss: 1.876
Epoch 30: Loss: 1.634
Epoch 40: Loss: 1.542
Epoch 50: Loss: 1.489  (converged)
```

### Step 4.3: Save Checkpoints

The script automatically saves:
- `results/{config_name}/model.pt` - Best model (lowest validation loss)
- `results/{config_name}/model_epoch_XX.pt` - Per-epoch checkpoints

---

## Phase 5: Evaluate Model (Days 2-3)

### Step 5.1: Create Evaluation Script

```python
# Script: evaluate_us_maritime.py
import torch
import numpy as np
import pickle
from TRAIS_Former_.CEE_TrAISformer.models import TrAISformer
from TRAIS_Former_.CEE_TrAISformer.config_us_maritime import Config
from TRAIS_Former_.CEE_TrAISformer import trainers

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in nautical miles"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance_km = R * c
    distance_nm = distance_km * 0.539957  # Convert to nautical miles
    
    return distance_nm

def evaluate():
    cf = Config()
    device = cf.device
    
    # Load model
    print("Loading model...")
    model = TrAISformer(cf)
    model.load_state_dict(torch.load(cf.ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    with open(cf.datadir + cf.testset_name, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Test set: {len(test_data)} trajectories")
    
    # Evaluate at different horizons
    horizons = [1, 3, 6, 10]  # hours
    errors_by_horizon = {h: [] for h in horizons}
    
    with torch.no_grad():
        for traj_idx, traj_data in enumerate(test_data[:100]):  # Test on 100 trajectories
            traj = torch.tensor(traj_data['traj'], dtype=torch.float32)
            
            if len(traj) < cf.init_seqlen + max(horizons) * 10:
                continue  # Skip if too short
            
            # Initial context
            context = traj[:cf.init_seqlen].unsqueeze(0).to(device)
            
            # Predict for each horizon
            for horizon in horizons:
                n_steps = int(horizon * 10)  # 10 points per hour
                
                # Generate predictions
                pred_traj = trainers.sample(
                    model, context, steps=n_steps,
                    sample_mode="pos_vicinity", r_vicinity=40
                )
                
                # Get predicted position
                pred_lat_norm = pred_traj[0, -1, 0].item()
                pred_lon_norm = pred_traj[0, -1, 1].item()
                
                # Convert back to degrees
                pred_lat = pred_lat_norm * (cf.lat_max - cf.lat_min) + cf.lat_min
                pred_lon = pred_lon_norm * (cf.lon_max - cf.lon_min) + cf.lon_min
                
                # Get ground truth
                actual_idx = cf.init_seqlen + n_steps
                if actual_idx < len(traj):
                    actual_lat_norm = traj[actual_idx, 0].item()
                    actual_lon_norm = traj[actual_idx, 1].item()
                    
                    actual_lat = actual_lat_norm * (cf.lat_max - cf.lat_min) + cf.lat_min
                    actual_lon = actual_lon_norm * (cf.lon_max - cf.lon_min) + cf.lon_min
                    
                    # Compute error
                    error_nm = haversine_distance(
                        actual_lat, actual_lon,
                        pred_lat, pred_lon
                    )
                    errors_by_horizon[horizon].append(error_nm)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for horizon in horizons:
        errors = errors_by_horizon[horizon]
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            print(f"{horizon:2d}h: {mean_error:6.2f} ± {std_error:5.2f} nautical miles")
    
    print("="*60)
    print("Target: < 10 nm at 10 hours")

if __name__ == "__main__":
    evaluate()
```

### Step 5.2: Run Evaluation

```bash
python evaluate_us_maritime.py
```

**Expected output:**
```
============================================================
EVALUATION RESULTS
============================================================
 1h:   0.45 ±  0.32 nautical miles
 3h:   1.23 ±  0.89 nautical miles
 6h:   2.56 ±  1.45 nautical miles
10h:   5.34 ±  2.67 nautical miles
============================================================
Target: < 10 nm at 10 hours
```

### Step 5.3: Generate Comparison Plots

```python
# Add to evaluate_us_maritime.py:
import matplotlib.pyplot as plt

# After computing errors, add:
fig, ax = plt.subplots(figsize=(10, 6))

horizons = sorted(errors_by_horizon.keys())
means = [np.mean(errors_by_horizon[h]) for h in horizons]
stds = [np.std(errors_by_horizon[h]) for h in horizons]

ax.errorbar(horizons, means, yerr=stds, marker='o', capsize=5, label='TrAISformer')
ax.axhline(y=10, color='r', linestyle='--', label='Target (10 nm)')
ax.set_xlabel('Prediction Horizon (hours)')
ax.set_ylabel('Position Error (nautical miles)')
ax.set_title('TrAISformer Performance on U.S. Maritime Data')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)
print("✓ Saved plot to evaluation_results.png")
```

---

## Troubleshooting Guide

### Issue: Out of Memory (OOM)
**Solution:**
```python
# In config: reduce batch_size
batch_size = 16  # Instead of 32

# Or reduce max_seqlen
max_seqlen = 60  # Instead of 120

# Or reduce embedding dimensions
n_lat_embd = 128  # Instead of 256
```

### Issue: NaN Loss During Training
**Solution:**
```python
# Check data normalization:
# All features should be in [0, 0.9999]

# Verify in prep_us_maritime.py:
traj_array[:, :4] = np.clip(traj_array[:, :4], 0, 0.9999)

# If still NaN, reduce learning rate:
learning_rate = 3e-4  # Instead of 6e-4
```

### Issue: Poor Prediction Accuracy
**Solution:**
```python
# 1. Increase training data
#    Need 500K+ trajectories for good U.S. model

# 2. Train longer
max_epochs = 100  # Instead of 50

# 3. Check data preprocessing
#    Verify timestamps are sequential
#    Verify SOG/COG values make sense

# 4. Tune sampling mode
sample_mode = "pos"  # Instead of "pos_vicinity"
#  for less constrained predictions
```

### Issue: Takes Too Long to Train
**Solution:**
```python
# 1. Use GPU (not CPU)
device = torch.device("cuda:0")

# 2. Reduce model size
n_layer = 4  # Instead of 8
n_head = 4   # Instead of 8

# 3. Reduce sequence length
max_seqlen = 60  # Instead of 120

# 4. Reduce dataset size (for testing)
n_train = 5000  # Instead of 50000
```

---

## Quick Command Reference

```bash
# Phase 1: Analyze data
python find_bounds.py

# Phase 2: Preprocess
python prep_us_maritime.py

# Phase 3: Configure
# Edit TRAIS_Former_/CEE_TrAISformer/config_trAISformer.py

# Phase 4: Train
cd TRAIS_Former_/CEE_TrAISformer
python trAISformer.py

# Phase 5: Evaluate
python evaluate_us_maritime.py

# Monitor GPU usage (if on Linux/WSL)
watch -n 1 nvidia-smi

# Kill training if needed
# Ctrl+C or:
pkill -f trAISformer.py
```

---

## Expected Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU Memory | 8 GB | 16 GB+ |
| GPU | V100 | A100 |
| CPU Cores | 4 | 8+ |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB | 200 GB |
| Training Time | 200 hours | 100 hours |

---

## Next Steps After Training

1. **Save the model**
   ```python
   torch.save(model.state_dict(), 'us_maritime_final_model.pt')
   ```

2. **Deploy for predictions**
   ```python
   # Create inference pipeline for real-time predictions
   ```

3. **Compare with baselines**
   ```python
   # LSTM, GRU, simple extrapolation for comparison
   ```

4. **Analyze failure cases**
   ```python
   # Which vessel types/regions have high error?
   # Any seasonal patterns?
   ```

5. **Publish results**
   ```
   - Model architecture: TrAISformer (Nguyen & Fablet, 2021)
   - Dataset: U.S. Maritime AIS data, 2020
   - Metrics: <X nm at 10-hour horizon
   - Improvement over baseline: +Y%
   ```
