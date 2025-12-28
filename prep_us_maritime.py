"""
Script to convert U.S. Maritime AIS data to TrAISformer format
Processes interpolated data from processed_data/*.csv and creates pickle files
"""

import numpy as np
import pandas as pd
import pickle
import glob
import os
from tqdm import tqdm
from pathlib import Path

# ============================================================================
# CONFIGURATION - ADJUST BASED ON YOUR U.S. MARITIME REGION
# ============================================================================

# Geographic bounds - analyze your data first!
LAT_MIN = None  # Will be computed or set manually
LAT_MAX = None
LON_MIN = None  # Will be computed or set manually
LON_MAX = None
SOG_MAX = 30.0  # Max speed in knots (typical for AIS)

# Data quality filters
MIN_SOG_THRESHOLD = 0.05  # knots - filter stationary vessels
MIN_TRAJECTORY_LENGTH = 36  # timesteps - minimum sequence length

# File paths
INPUT_CSV_PATTERN = 'processed_data/AIS_*.csv'
OUTPUT_DIR = 'TRAIS_Former_/CEE_TrAISformer/data/us_maritime'
TIMESTAMP_COLUMN = 'BaseDateTime'
MMSI_COLUMN = 'MMSI'
LAT_COLUMN = 'LAT'
LON_COLUMN = 'LON'
SOG_COLUMN = 'SOG'
COG_COLUMN = 'COG'

# ============================================================================
# STEP 1: ANALYZE DATA BOUNDS
# ============================================================================

def analyze_data_bounds(csv_files):
    """Analyze geographic and velocity bounds of your data"""
    print("Analyzing data bounds from CSV files...")
    
    lats = []
    lons = []
    sogs = []
    
    for csv_file in tqdm(csv_files):
        try:
            df_chunk = pd.read_csv(csv_file)
            lats.extend(df_chunk[LAT_COLUMN].dropna().values)
            lons.extend(df_chunk[LON_COLUMN].dropna().values)
            sogs.extend(df_chunk[SOG_COLUMN].dropna().values)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue
    
    lats = np.array(lats)
    lons = np.array(lons)
    sogs = np.array(sogs)
    
    results = {
        'lat_min': np.nanmin(lats),
        'lat_max': np.nanmax(lats),
        'lon_min': np.nanmin(lons),
        'lon_max': np.nanmax(lons),
        'sog_min': np.nanmin(sogs),
        'sog_max': np.nanmax(sogs),
        'n_points': len(lats)
    }
    
    print("\n" + "="*60)
    print("DATA BOUNDS ANALYSIS")
    print("="*60)
    print(f"Latitude:  {results['lat_min']:.2f}° to {results['lat_max']:.2f}°")
    print(f"Longitude: {results['lon_min']:.2f}° to {results['lon_max']:.2f}°")
    print(f"SOG:       {results['sog_min']:.2f} to {results['sog_max']:.2f} knots")
    print(f"Total points: {results['n_points']:,}")
    print("="*60)
    print("\n⚠️  USE THESE VALUES TO SET LAT_MIN, LAT_MAX, LON_MIN, LON_MAX ABOVE")
    print("   WITH APPROPRIATE PADDING (e.g., ±1 degree from bounds)\n")
    
    return results


# ============================================================================
# STEP 2: LOAD AND CONCATENATE ALL CSV FILES
# ============================================================================

def load_ais_data(csv_pattern):
    """Load all CSV files matching pattern"""
    print(f"Loading CSV files matching: {csv_pattern}")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {csv_pattern}")
    
    print(f"Found {len(csv_files)} files")
    
    df_list = []
    for csv_file in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Ensure timestamp column is datetime
    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    
    # Sort by vessel and time
    df = df.sort_values([MMSI_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df):,} total AIS records")
    print(f"✓ Found {df[MMSI_COLUMN].nunique():,} unique vessels")
    
    return df


# ============================================================================
# STEP 3: PREPROCESS TRAJECTORIES
# ============================================================================

def preprocess_trajectories(df, lat_min, lat_max, lon_min, lon_max, sog_max):
    """
    Convert raw AIS data to TrAISformer format
    
    Output format:
    [
        {
            "mmsi": vessel_id,
            "traj": numpy array (N, 5)
                    columns: [LAT_norm, LON_norm, SOG_norm, COG_norm, TIMESTAMP_unix]
        },
        ...
    ]
    """
    
    trajectories = []
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    print(f"\nProcessing {df[MMSI_COLUMN].nunique():,} vessels...")
    
    for mmsi, group in tqdm(df.groupby(MMSI_COLUMN), desc="Processing vessels"):
        group = group.reset_index(drop=True)
        
        # FILTER 1: Remove stationary vessels (SOG > threshold)
        moving_mask = group[SOG_COLUMN] > MIN_SOG_THRESHOLD
        if moving_mask.sum() == 0:
            continue  # Entire trajectory is stationary
        
        # Find first moving point
        first_moving_idx = moving_mask.idxmax()
        group = group[first_moving_idx:].reset_index(drop=True)
        
        # FILTER 2: Remove NaN values
        required_cols = [LAT_COLUMN, LON_COLUMN, SOG_COLUMN, COG_COLUMN, TIMESTAMP_COLUMN]
        group = group.dropna(subset=required_cols).reset_index(drop=True)
        
        if len(group) < MIN_TRAJECTORY_LENGTH:
            continue  # Trajectory too short
        
        # FILTER 3: Remove outliers (out of geographic bounds)
        in_bounds = (
            (group[LAT_COLUMN] >= lat_min - 1) & (group[LAT_COLUMN] <= lat_max + 1) &
            (group[LON_COLUMN] >= lon_min - 1) & (group[LON_COLUMN] <= lon_max + 1)
        )
        group = group[in_bounds].reset_index(drop=True)
        
        if len(group) < MIN_TRAJECTORY_LENGTH:
            continue
        
        # BUILD TRAJECTORY ARRAY
        traj_array = np.zeros((len(group), 5), dtype=np.float32)
        
        # Normalize latitude to [0, 1)
        traj_array[:, 0] = (group[LAT_COLUMN].values - lat_min) / lat_range
        
        # Normalize longitude to [0, 1)
        traj_array[:, 1] = (group[LON_COLUMN].values - lon_min) / lon_range
        
        # Normalize SOG to [0, 1)
        traj_array[:, 2] = group[SOG_COLUMN].values / sog_max
        
        # Normalize COG (0-360°) to [0, 1)
        traj_array[:, 3] = group[COG_COLUMN].values / 360.0
        
        # Convert timestamp to Unix seconds
        traj_array[:, 4] = group[TIMESTAMP_COLUMN].astype(np.int64).values // 10**9
        
        # CLIP to [0, 0.9999) - prevents boundary issues
        traj_array[:, :4] = np.clip(traj_array[:, :4], 0, 0.9999)
        
        trajectories.append({
            "mmsi": int(mmsi),
            "traj": traj_array
        })
    
    print(f"\n✓ Extracted {len(trajectories):,} valid trajectories")
    
    return trajectories


# ============================================================================
# STEP 4: TRAIN/VAL/TEST SPLIT
# ============================================================================

def train_val_test_split(trajectories, train_ratio=0.8, val_ratio=0.1):
    """
    Split trajectories by vessel (not by sequence)
    This prevents temporal leakage
    """
    n_vessels = len(trajectories)
    n_train = int(train_ratio * n_vessels)
    n_val = int(val_ratio * n_vessels)
    
    print(f"\nTrain/Val/Test Split (80/10/10 by vessel):")
    print(f"  Train: {n_train:,} vessels")
    print(f"  Val:   {n_val:,} vessels")
    print(f"  Test:  {n_vessels - n_train - n_val:,} vessels")
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:n_train + n_val]
    test_data = trajectories[n_train + n_val:]
    
    return train_data, val_data, test_data


# ============================================================================
# STEP 5: SAVE AS PICKLE
# ============================================================================

def save_pickle_files(train_data, val_data, test_data, output_dir):
    """Save data as pickle files for TrAISformer"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save files
    files_info = [
        (train_data, 'us_maritime_train.pkl'),
        (val_data, 'us_maritime_valid.pkl'),
        (test_data, 'us_maritime_test.pkl'),
    ]
    
    print(f"\nSaving pickle files to {output_dir}/")
    
    for data, filename in files_info:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Calculate statistics
        n_trajectories = len(data)
        total_timesteps = sum(len(traj['traj']) for traj in data)
        avg_length = total_timesteps / n_trajectories if n_trajectories > 0 else 0
        
        print(f"  ✓ {filename}")
        print(f"    - {n_trajectories:,} trajectories")
        print(f"    - {total_timesteps:,} total timesteps")
        print(f"    - {avg_length:.1f} avg length")


# ============================================================================
# STEP 6: GENERATE CONFIG
# ============================================================================

def generate_config_template(lat_min, lat_max, lon_min, lon_max):
    """Generate configuration for TrAISformer"""
    
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    config_text = f'''
# TrAISformer Configuration for U.S. Maritime Data
# Auto-generated from preprocessing script

import os
import torch

class Config():
    # Device and training
    retrain = True
    device = torch.device("cuda:0")
    max_epochs = 50
    batch_size = 32
    
    # Sequence lengths
    init_seqlen = 18      # Initial sequence for conditioning
    max_seqlen = 120      # Maximum sequence length
    min_seqlen = 36       # Minimum sequence for training
    
    dataset_name = "us_maritime"
    
    # U.S. Maritime Data Configuration
    if dataset_name == "us_maritime":
        # Geographic bounds (computed from data)
        lat_min = {lat_min:.2f}
        lat_max = {lat_max:.2f}
        lon_min = {lon_min:.2f}
        lon_max = {lon_max:.2f}
        
        # Feature bin sizes (quantization levels)
        # Larger regions need more bins
        lat_size = int(({lat_range:.2f}) * 10) + 10  # ~{int(lat_range * 10) + 10}
        lon_size = int(({lon_range:.2f}) * 10) + 10  # ~{int(lon_range * 10) + 10}
        sog_size = 30   # 0-30 knots
        cog_size = 72   # 0-360 degrees (5° per bin)
        
        # Embedding dimensions
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
    
    # Model architecture
    n_head = 8          # Number of attention heads
    n_layer = 8         # Number of transformer blocks
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    
    # Dropout
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # Optimization
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    
    # Data paths
    datadir = "./data/us_maritime/"
    trainset_name = "us_maritime_train.pkl"
    validset_name = "us_maritime_valid.pkl"
    testset_name = "us_maritime_test.pkl"
    
    # Model parameters
    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10
    r_vicinity = 40
    
    # Blur settings
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    
    # Checkpoint
    savedir = "./results/us_maritime/"
    ckpt_path = os.path.join(savedir, "model.pt")
    
    num_workers = 4
'''
    
    return config_text.strip()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("TrAISformer Data Preprocessing for U.S. Maritime")
    print("="*70)
    
    # Step 1: Find CSV files
    csv_files = sorted(glob.glob(INPUT_CSV_PATTERN))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {INPUT_CSV_PATTERN}")
    
    # Step 1b: Analyze bounds (REQUIRED FIRST TIME)
    print("\n[STEP 1] Analyzing data bounds...")
    bounds = analyze_data_bounds(csv_files)
    
    # MANUAL CONFIGURATION NEEDED
    print("\n⚠️  IMPORTANT: Update LAT_MIN, LAT_MAX, LON_MIN, LON_MAX at top of script")
    print("    Then re-run this script.")
    
    if LAT_MIN is None or LAT_MAX is None or LON_MIN is None or LON_MAX is None:
        print("\n[SKIPPED] Geographic bounds not configured yet.")
        print("         Configure bounds above and re-run.")
        return
    
    # Step 2: Load data
    print("\n[STEP 2] Loading AIS data...")
    df = load_ais_data(INPUT_CSV_PATTERN)
    
    # Step 3: Preprocess
    print("\n[STEP 3] Preprocessing trajectories...")
    trajectories = preprocess_trajectories(df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SOG_MAX)
    
    # Step 4: Split
    print("\n[STEP 4] Splitting data...")
    train_data, val_data, test_data = train_val_test_split(trajectories)
    
    # Step 5: Save
    print("\n[STEP 5] Saving pickle files...")
    save_pickle_files(train_data, val_data, test_data, OUTPUT_DIR)
    
    # Step 6: Generate config
    print("\n[STEP 6] Generating configuration template...")
    config_text = generate_config_template(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    
    config_path = os.path.join(OUTPUT_DIR, '../../../config_us_maritime.py')
    with open(config_path, 'w') as f:
        f.write(config_text)
    
    print(f"\n✓ Configuration template saved to: {config_path}")
    print("  Review and copy relevant sections to config_trAISformer.py")
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"""
Next steps:
1. Review generated config in: {config_path}
2. Update TRAIS_Former_/CEE_TrAISformer/config_trAISformer.py with US settings
3. Run training:
   python TRAIS_Former_/CEE_TrAISformer/trAISformer.py
    """)


if __name__ == "__main__":
    main()
