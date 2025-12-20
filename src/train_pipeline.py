# src/train_pipeline.py
import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from src.data_loader import VesselSequenceDataset
from src.models import model_factory
from src.utils import ensure_dir, compute_regression_metrics, save_metrics_csv, plot_loss_curves, plot_metrics

def train_pipeline(pkl_path, output_dir, input_features, target_features,
                   model_name='lstm', seq_len=10, pred_len=1,
                   batch_size=128, epochs=30, lr=1e-3, hidden_dim=128,
                   device=None, sequence_cache=None):

    ensure_dir(output_dir)
    df = joblib.load(pkl_path) if pkl_path.endswith('.pkl') else pd.read_pickle(pkl_path)

    dataset = VesselSequenceDataset(df, input_features, target_features,
                                    seq_len=seq_len, pred_len=pred_len,
                                    save_path=sequence_cache)

    # Train / Validation split
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model_factory(model_name, input_dim=len(input_features),
                          hidden_dim=hidden_dim, output_dim=len(target_features))
    model = model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_id = f"{model_name}_{int(time.time())}"
    run_dir = os.path.join(output_dir, 'runs', run_id)
    ensure_dir(run_dir)
    best_val_loss = float('inf')

    history = {'train_loss':[], 'val_loss':[], 'train_mse':[], 'val_mse':[], 'train_mae':[], 'val_mae':[]}

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_losses = []
        for X, y, _ in tqdm(train_loader, desc=f"Train epoch {epoch+1}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            if preds.ndim != y.ndim:
                y = y.squeeze(1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        all_pred, all_y = [], []
        with torch.no_grad():
            for Xv, yv, _ in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pv = model(Xv)
                if pv.ndim != yv.ndim:
                    yv = yv.squeeze(1)
                lossv = criterion(pv, yv)
                val_losses.append(lossv.item())
                all_pred.append(pv.cpu().numpy())
                all_y.append(yv.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        all_pred = np.concatenate(all_pred, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        metrics = compute_regression_metrics(all_y, all_pred)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(metrics['mse'])
        history['val_mse'].append(metrics['mse'])
        history['train_mae'].append(metrics['mae'])
        history['val_mae'].append(metrics['mae'])

        # checkpoint best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(output_dir, 'models', f"{run_id}_best.pth")
            ensure_dir(os.path.dirname(ckpt_path))
            torch.save({'model_state': model.state_dict(), 'config': {'model_name':model_name}}, ckpt_path)

    # save final model and history
    final_path = os.path.join(output_dir, 'models', f"{run_id}_final.pth")
    torch.save({'model_state': model.state_dict(), 'config': {'model_name':model_name}}, final_path)
    save_metrics_csv({i:{k:history[k][i] for k in history} for i in range(len(history['train_loss']))}, os.path.join(run_dir, 'history.csv'))
    plot_loss_curves(history, os.path.join(run_dir, 'loss_curve.png'))
    plot_metrics(history, 'mse', os.path.join(run_dir, 'mse_curve.png'))

    print("✅ Training completed. Run saved at:", run_dir)
    return final_path, run_dir
