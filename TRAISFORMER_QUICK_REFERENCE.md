# TrAISformer Quick Reference - Algorithms & Features Summary

## Paper at a Glance

| Aspect | Details |
|--------|---------|
| **Title** | TrAISformer -- A Transformer Network with Sparse Augmented Data Representation and Cross Entropy Loss for AIS-based Vessel Trajectory Prediction |
| **Authors** | Duong Nguyen, Ronan Fablet |
| **Year** | 2021 (updated 2024) |
| **Main Contribution** | Novel discrete representation of AIS data + cross-entropy loss for multimodal trajectory prediction |
| **Performance** | <10 nautical miles error up to 10 hours ahead |
| **Baseline Dataset** | Danish Maritime Authority (DMA) - Baltic/North Sea |

---

## Algorithm Core Components

### 1. INPUT REPRESENTATION
```
Raw AIS Features (4D)
├─ Latitude (0-360°)
├─ Longitude (0-360°) 
├─ SOG - Speed Over Ground (knots)
└─ COG - Course Over Ground (degrees)

          ↓ Normalize to [0,1)

Continuous Vector (4D)
[0.5234, 0.3021, 0.67, 0.45]

          ↓ Discretize (Unique: TrAISformer's Key Innovation)

Discrete Tokens (4D indices)
[lat_idx: 130, lon_idx: 81, sog_idx: 20, cog_idx: 32]
```

### 2. TOKENIZATION/DISCRETIZATION (The Key Innovation)

```python
# Convert continuous values → categorical indices
lat_normalized = (lat - lat_min) / (lat_max - lat_min)      # [0, 1)
lat_token = int(lat_normalized * 250)                        # [0, 249]

lon_normalized = (lon - lon_min) / (lon_max - lon_min)      # [0, 1)
lon_token = int(lon_normalized * 270)                        # [0, 269]

sog_normalized = sog / 30                                    # [0, 1)
sog_token = int(sog_normalized * 30)                         # [0, 29]

cog_normalized = cog / 360                                   # [0, 1)
cog_token = int(cog_normalized * 72)                         # [0, 71]
```

**Why?** 
- Transforms regression → classification (easier for transformers)
- Enables cross-entropy loss (naturally handles multimodal predictions)
- Each dimension has independent vocab size

### 3. EMBEDDING PROJECTION

```python
# Lookup embeddings for each token
lat_emb = embedding_lat_table[lat_token]      # 256D
lon_emb = embedding_lon_table[lon_token]      # 256D
sog_emb = embedding_sog_table[sog_token]      # 128D
cog_emb = embedding_cog_table[cog_token]      # 128D

# Concatenate embeddings
token_embedding = concatenate([lat_emb, lon_emb, sog_emb, cog_emb])  # 768D

# Add positional encoding
token_with_pos = token_embedding + positional_embedding[timestep]    # 768D
```

**Architecture Details:**
- **Embedding Tables:** 4 separate lookup tables (not shared)
  - lat_table: shape (250, 256)
  - lon_table: shape (270, 256)
  - sog_table: shape (30, 128)
  - cog_table: shape (72, 128)
- **Positional Embeddings:** Learnable, shape (max_seqlen=120, 768)

### 4. TRANSFORMER BLOCKS

```
Input: (batch_size, seq_len, 768)
   ↓ Layer Norm
   ↓ Causal Self-Attention (8 heads)
      - Multi-head attention with causal masking
      - Prevents attending to future positions
   ↓ Residual connection
   ↓ Layer Norm
   ↓ Feed-Forward (768 → 3072 → 768)
      - Two linear layers + GELU activation
   ↓ Residual connection
   ↓ Repeat × 8 blocks
Output: (batch_size, seq_len, 768)
```

**Attention Mechanism:**
```
Query = X @ W_q    (batch, heads, seq_len, head_dim)
Key = X @ W_k
Value = X @ W_v

Attention = softmax(QK^T / sqrt(d)) @ V  [with causal mask]
```

### 5. CLASSIFICATION HEAD & LOSS

```python
# Linear projection to vocab sizes
logits = linear_head(transformer_output)  # (batch, seq, 622)
                                          # where 622 = 250+270+30+72

# Split logits
lat_logits, lon_logits, sog_logits, cog_logits = split(logits)
# lat_logits: (batch, seq, 250)
# lon_logits: (batch, seq, 270)
# sog_logits: (batch, seq, 30)
# cog_logits: (batch, seq, 72)

# Loss computation (CROSS-ENTROPY, not MSE!)
loss_lat = CrossEntropyLoss(lat_logits, lat_targets)
loss_lon = CrossEntropyLoss(lon_logits, lon_targets)
loss_sog = CrossEntropyLoss(sog_logits, sog_targets)
loss_cog = CrossEntropyLoss(cog_logits, cog_targets)

loss_total = loss_lat + loss_lon + loss_sog + loss_cog
```

**Why Cross-Entropy?**
- Handles multimodal distributions naturally
- Better gradient flow than MSE for classification
- Natural probability calibration via softmax

### 6. INFERENCE (Autoregressive Generation)

```
Initialization:
  context = first 18 timesteps of trajectory

Loop for desired forecast horizon:
  1. Forward pass through transformer
     logits = model(context)  # (batch, 18, 622)
  
  2. Extract last timestep logits
     logits = logits[:, -1, :]  # (batch, 622)
  
  3. Apply sampling (multiple strategies):
     
     a) "pos_vicinity" mode (constrained):
        - Only keep positions within r_vicinity of current position
        - Prevents unrealistic jumps
        - Applied to lat & lon logits
     
     b) Top-K filtering:
        - Keep only top K most likely values
        - Reduces unlikely predictions
     
  4. Convert logits → probabilities
     lat_probs = softmax(lat_logits)   # (batch, 250)
     lon_probs = softmax(lon_logits)   # (batch, 270)
     sog_probs = softmax(sog_logits)   # (batch, 30)
     cog_probs = softmax(cog_logits)   # (batch, 72)
  
  5. Sample or take argmax
     if sample:
        lat_idx = multinomial(lat_probs)      # Stochastic
     else:
        lat_idx = argmax(lat_probs)           # Deterministic
  
     (same for lon, sog, cog)
  
  6. Convert indices → real values
     lat_pred = lat_idx / 250 * (lat_max - lat_min) + lat_min
     lon_pred = lon_idx / 270 * (lon_max - lon_min) + lon_min
     sog_pred = sog_idx / 30
     cog_pred = cog_idx / 72 * 360
  
  7. Append to context
     context = [context, [lat_pred, lon_pred, sog_pred, cog_pred]]

Return: Full trajectory including predictions
```

---

## Data Preprocessing Pipeline

### Input Requirements
```
Raw CSV with columns:
- MMSI (vessel identifier)
- Timestamp (sortable)
- LAT (latitude in degrees)
- LON (longitude in degrees)
- SOG (speed in knots)
- COG (course in degrees, 0-360)
```

### Processing Steps

**Step 1: Load and Sort**
```python
df.sort_values(['MMSI', 'Timestamp'])  # Per-vessel chronological order
```

**Step 2: Group by Vessel**
```python
for mmsi, vessel_trajectory in df.groupby('MMSI'):
    # Process each vessel independently
```

**Step 3: Filter Moving Vessels**
```python
moving_indices = vessel_traj['SOG'] > 0.05  # knots
vessel_traj = vessel_traj[moving_indices]

# Use first moving point as trajectory start
first_moving_idx = moving_indices.idxmax()
vessel_traj = vessel_traj[first_moving_idx:]
```

**Step 4: Remove Bad Data**
```python
# Remove NaN values
vessel_traj = vessel_traj.dropna(subset=['LAT', 'LON', 'SOG', 'COG', 'TIMESTAMP'])

# Remove if too short
if len(vessel_traj) < 36:
    skip_vessel()

# Remove if out of bounds
in_bounds = (
    (LAT >= LAT_MIN - 1) & (LAT <= LAT_MAX + 1) &
    (LON >= LON_MIN - 1) & (LON <= LON_MAX + 1)
)
vessel_traj = vessel_traj[in_bounds]
```

**Step 5: Normalize Features**
```python
# Geographic normalization (region-specific)
LAT_norm = (LAT - LAT_MIN) / (LAT_MAX - LAT_MIN)       # [0, 1)
LON_norm = (LON - LON_MIN) / (LON_MAX - LON_MIN)       # [0, 1)

# Speed normalization
SOG_norm = SOG / 30.0                                   # [0, 1) for 30 knot max

# Course normalization
COG_norm = COG / 360.0                                  # [0, 1)

# Clip to [0, 0.9999] to avoid boundary issues
all_features = np.clip(all_features, 0, 0.9999)
```

**Step 6: Create Output Format**
```python
# Save as list of dictionaries
trajectories = [
    {
        "mmsi": 123456789,
        "traj": np.array([
            [0.5234, 0.3021, 0.67, 0.45, 1704067200],  # timestep 0
            [0.5245, 0.3032, 0.68, 0.45, 1704067560],  # timestep 1 (+10 min)
            [...],
        ])  # shape: (N, 5) where N is sequence length
    },
    {...},
]

# Columns: [LAT_norm, LON_norm, SOG_norm, COG_norm, TIMESTAMP_unix]
```

**Step 7: Train/Val/Test Split**
```python
# Split by MMSI (vessel), NOT by timestep!
# This prevents temporal leakage

n_vessels = len(trajectories)
n_train = int(0.8 * n_vessels)
n_val = int(0.1 * n_vessels)

train_data = trajectories[:n_train]           # 80%
val_data = trajectories[n_train:n_train+n_val]           # 10%
test_data = trajectories[n_train+n_val:]     # 10%

# Save each as separate pickle file
pickle.dump(train_data, open('train.pkl', 'wb'))
pickle.dump(val_data, open('valid.pkl', 'wb'))
pickle.dump(test_data, open('test.pkl', 'wb'))
```

---

## Training Details

### Data Loading
```python
# Each batch item:
seq:       (max_seqlen=120, 4) - normalized trajectory
mask:      (max_seqlen,) - 0 for padding, 1 for real data
seqlen:    scalar - actual sequence length
mmsi:      scalar - vessel identifier
time_start: scalar - Unix timestamp of first point
```

### Training Loop
```python
for epoch in range(max_epochs):
    for batch in train_loader:
        seq, mask, seqlen, mmsi, time_start = batch
        
        # Shift: input = seq[:-1], target = seq[1:]
        # This makes it predict the next timestep
        inputs = seq[:, :-1, :]       # (batch, 119, 4)
        targets = seq[:, 1:, :]       # (batch, 119, 4)
        
        # Forward pass
        logits, _ = model(inputs)     # (batch, 119, 622)
        
        # Compute loss (4 cross-entropy terms)
        loss = compute_loss(logits, targets, model)
        
        # Backward + optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Evaluation Metrics

**Haversine Distance (Ground Truth Metric)**
```python
def haversine(lat1, lon1, lat2, lon2):
    """Distance in kilometers"""
    R = 6371  # Earth radius
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c  # in km

# Convert to nautical miles
distance_nm = distance_km * 0.539957
```

**Evaluation Protocol**
```python
# Test on 1h, 3h, 6h, 10h horizons
# Assuming 6-minute sampling (10 points/hour):

errors = {}
for horizon_hours in [1, 3, 6, 10]:
    n_steps = horizon_hours * 10
    
    # Generate predictions
    predictions = sample(model, context, steps=n_steps, 
                        sample_mode="pos_vicinity")
    
    # Compare to ground truth
    error_nm = haversine(
        actual_lat, actual_lon,
        pred_lat, pred_lon
    ) * 0.539957
    
    errors[horizon_hours] = error_nm.mean()

# Expected: < 10 nm at 10 hours
```

---

## Key Implementation Decisions for U.S. Data

| Decision | Denmark (Original) | U.S. (Your Implementation) |
|----------|-------------------|-----------------------|
| **Geographic Bounds** | lat: 55.5-58°N lon: 10.3-13°E | Analyze from your data |
| **LAT bins** | 250 | ≈ (lat_max - lat_min) × 10 |
| **LON bins** | 270 | ≈ (lon_max - lon_min) × 10 |
| **Training Data** | ~100K trajectories | Target: 500K+ trajectories |
| **Region Seasonality** | Moderate | High (adjust if needed) |
| **Vessel Types** | Mixed maritime | May be more diverse |
| **GPU Time** | ~100 hours | ~200 hours (larger region) |

---

## Common Issues & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss not decreasing | Bad normalization bounds | Re-analyze geographic min/max |
| Predictions at boundary | Incorrect clipping | Ensure features clipped to [0, 0.9999) |
| Unrealistic jumps | No "pos_vicinity" filtering | Enable constrained sampling |
| Poor long-horizon accuracy | Model overfitting | Increase dropout, reduce lr |
| Out of memory | Large batch size + long sequences | Reduce batch_size or max_seqlen |
| High spatial errors | Wrong CoG bins | Verify cog_size=72 (5° per bin) |

---

## Files to Create/Modify

```
Your Workspace:
├── TRAIS_FORMER_REPLICATION_GUIDE.md       ✓ Created (detailed guide)
├── prep_us_maritime.py                     ✓ Created (preprocessing script)
│
└── TRAIS_Former_/CEE_TrAISformer/
    ├── config_trAISformer.py               ← MODIFY (geographic bounds)
    ├── data/us_maritime/
    │   ├── us_maritime_train.pkl           ← Generate using prep_us_maritime.py
    │   ├── us_maritime_valid.pkl
    │   └── us_maritime_test.pkl
    ├── trAISformer.py                      ✓ No changes needed
    ├── models.py                           ✓ No changes needed
    ├── datasets.py                         ✓ No changes needed
    ├── trainers.py                         ✓ No changes needed
    └── utils.py                            ✓ No changes needed
```

---

## Execution Timeline

| Phase | Time | Tasks |
|-------|------|-------|
| **Phase 1: Data Prep** | 2-3 days | Analyze bounds, create preprocessing script, generate pickle files |
| **Phase 2: Configuration** | 1 day | Update config with geographic bounds |
| **Phase 3: Training** | 7-14 days | Train on GPU (100-200 GPU hours total) |
| **Phase 4: Evaluation** | 2-3 days | Compute metrics, analyze results |
| **Total** | ~3 weeks | (can be parallelized) |

---

## References

- **Paper:** arXiv:2109.03958v4
- **Code:** https://github.com/CIA-Oceanix/trAISformer
- **Related:** GeoTrackNet (preprocessing baseline)
- **Transformer:** minGPT architecture (https://github.com/karpathy/minGPT)

