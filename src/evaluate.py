# src/evaluate.py
import os, numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.utils import ensure_dir, compute_regression_metrics
from sklearn.externals import joblib  # or import joblib

def evaluate_model(checkpoint_path, df, input_features, target_features, seq_len=10, pred_len=1, device=None, batch_size=256, output_dir='outputs'):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    ckpt = torch.load(checkpoint_path, map_location=device)
    from src.models import model_factory
    model_name = ckpt['config']['model_name'] if 'config' in ckpt else 'lstm'
    model = model_factory(model_name, input_dim=len(input_features))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    # Build dataset similar to training (or reuse saved test loader)
    from src.data_loader import VesselSequenceDataset
    dataset = VesselSequenceDataset(df, input_features, target_features, seq_len=seq_len, pred_len=pred_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds=[]; all_y=[]; all_mmsi=[]
    with torch.no_grad():
        for X,y,mmsi in tqdm(loader, desc='Evaluating'):
            X = X.to(device); y = y.to(device)
            p = model(X)
            all_preds.append(p.cpu().numpy()); all_y.append(y.cpu().numpy()); all_mmsi.extend(mmsi)
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # inverse scale must be done by caller (pass scalers if needed)
    metrics = compute_regression_metrics(all_y, all_preds)
    print("Global metrics (on scaled outputs):", metrics)

    # Per-MMSI aggregation (example: RMSE on lat/lon)
    df_res = pd.DataFrame({
        'MMSI': all_mmsi
    })
    for i, col in enumerate(target_features):
        df_res[f'{col}_true'] = all_y[:, i]
        df_res[f'{col}_pred'] = all_preds[:, i]
        df_res[f'{col}_abs_err'] = np.abs(all_y[:, i] - all_preds[:, i])

    per_mmsi = df_res.groupby('MMSI')[[c for c in df_res.columns if '_abs_err' in c]].mean()
    ensure_dir(output_dir)
    per_mmsi.to_csv(os.path.join(output_dir, 'per_mmsi_metrics.csv'))
    df_res.to_csv(os.path.join(output_dir, 'predictions_detailed.csv'), index=False)
    return metrics, per_mmsi
