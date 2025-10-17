# src/train.py
import os, time, argparse, json
import numpy as np, pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib

from src.utils import ensure_dir, compute_regression_metrics, save_metrics_csv, plot_loss_curves, plot_metrics
from src.data_loader import VesselSequenceDataset
from src.models import model_factory

def train_pipeline(pkl_path, output_dir, input_features, target_features, model_name='lstm',
                   seq_len=10, pred_len=1, batch_size=128, epochs=30, lr=1e-3, hidden_dim=128, device=None):
    ensure_dir(output_dir)
    df = pd.read_pickle(pkl_path)
    # assume df already cleaned and scaled; else incorporate scalers here
    dataset = VesselSequenceDataset(df, input_features, target_features, seq_len=seq_len, pred_len=pred_len)
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = model_factory(model_name, input_dim=len(input_features), hidden_dim=hidden_dim, output_dim=len(target_features))
    model = model.to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_id = f"{model_name}_{int(time.time())}"
    run_dir = os.path.join(output_dir, 'runs', run_id)
    ensure_dir(run_dir)
    writer = SummaryWriter(run_dir)

    history = {'train_loss':[], 'val_loss':[], 'train_mse':[], 'val_mse':[], 'train_mae':[], 'val_mae':[]}

    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        train_losses = []
        for X, y, _ in tqdm(train_loader, desc=f"Train epoch {epoch+1}", leave=False):
            X = X.to(device); y = y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # validation
        model.eval()
        val_losses = []; all_pred=[]; all_y=[]
        with torch.no_grad():
            for Xv, yv, _ in tqdm(val_loader, desc=f"Val epoch {epoch+1}", leave=False):
                Xv = Xv.to(device); yv = yv.to(device)
                pv = model(Xv)
                lossv = criterion(pv, yv)
                val_losses.append(lossv.item())
                all_pred.append(pv.cpu().numpy()); all_y.append(yv.cpu().numpy())
        avg_val_loss = np.mean(val_losses)
        all_pred = np.concatenate(all_pred, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        # per-epoch metrics on raw outputs (still scaled) — user may inverse-scale later for meters
        metrics_train = compute_regression_metrics(all_y, all_pred) if len(all_pred)>0 else {'mse':np.nan,'mae':np.nan,'rmse':np.nan,'r2':np.nan}
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(metrics_train['mse'])
        history['val_mse'].append(metrics_train['mse'])  # same here because we lack train preds separately
        history['train_mae'].append(metrics_train['mae'])
        history['val_mae'].append(metrics_train['mae'])

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('MSE/val', metrics_train['mse'], epoch)

        # checkpoint best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(output_dir, 'models', f"{run_id}_best.pth")
            ensure_dir(os.path.dirname(ckpt_path))
            torch.save({'model_state': model.state_dict(), 'config': {'model_name':model_name, 'input_features':input_features, 'target_features':target_features}}, ckpt_path)

    # save final model and history
    final_path = os.path.join(output_dir, 'models', f"{run_id}_final.pth")
    torch.save({'model_state': model.state_dict(), 'config': {'model_name':model_name, 'input_features':input_features, 'target_features':target_features}}, final_path)
    save_metrics_csv({i:{k:history[k][i] for k in history} for i in range(len(history['train_loss']))}, os.path.join(run_dir, 'history.csv'))
    plot_loss_curves(history, os.path.join(run_dir, 'loss_curve.png'))
    plot_metrics(history, 'mse', os.path.join(run_dir, 'mse_curve.png'))
    writer.close()
    print("Run saved to:", run_dir)
    return final_path, run_dir





def main():
    parser = argparse.ArgumentParser(description="Train vessel trajectory prediction models (LSTM / TCN).")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to processed AIS .pkl file.")
    parser.add_argument("--output_dir", type=str, default="outputs_pipeline", help="Directory for logs/models/plots.")
    parser.add_argument("--model_name", type=str, default="lstm", choices=["lstm", "tcn"], help="Model architecture to train.")
    parser.add_argument("--seq_len", type=int, default=10, help="Input sequence length.")
    parser.add_argument("--pred_len", type=int, default=1, help="Prediction length.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    # Define your input & target feature sets (should match your data preprocessing)
    input_features = ["LAT", "LON", "SOG", "COG", "Heading", "v_x", "v_y",
                      "turn_rate", "accel_knots_per_hr", "dayofweek", "month",
                      "Δt_hours", "ΔCOG", "ΔSOG", "COG_rad"]
    target_features = ["LAT", "LON", "SOG", "COG", "Heading"]

    print(f"🚀 Training {args.model_name.upper()} model for {args.epochs} epochs...")
    print(f"📦 Using dataset: {args.pkl_path}")
    print(f"💾 Outputs will be saved in: {args.output_dir}")

    final_model_path, run_dir = train_pipeline(
        pkl_path=args.pkl_path,
        output_dir=args.output_dir,
        input_features=input_features,
        target_features=target_features,
        model_name=args.model_name,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim
    )

    print(f"✅ Training finished. Model saved at: {final_model_path}")
    print(f"📈 Run logs and metrics available in: {run_dir}")













if __name__ == "__main__":
    main()


