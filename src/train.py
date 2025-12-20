import os, time, argparse, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.data_loader import LazyVesselSequenceDataset
from src.models import model_factory
from src.utils import ensure_dir, compute_regression_metrics, save_metrics_csv, plot_loss_curves, plot_metrics

def train_pipeline(args):
    ensure_dir(args.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_pickle(args.pkl_path)

    input_features = ["LAT", "LON", "SOG", "COG", "Heading", "v_x", "v_y",
                      "turn_rate", "accel_knots_per_hr", "dayofweek", "month",
                      "Δt_hours", "ΔCOG", "ΔSOG", "COG_rad"]
    target_features = ["LAT", "LON", "SOG", "COG", "Heading"]

    dataset = LazyVesselSequenceDataset(df, input_features, target_features,
                                        seq_len=args.seq_len, pred_len=args.pred_len,
                                        device=device)

    # DataLoader with lazy batching
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)
    

    # Model
    model = model_factory(args.model_name, input_dim=len(input_features),
                          hidden_dim=args.hidden_dim, output_dim=len(target_features)).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    run_id = f"{args.model_name}_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, 'runs', run_id)
    ensure_dir(run_dir)
    writer = SummaryWriter(run_dir)

    best_val_loss = float('inf')
    history = {'train_loss':[], 'val_loss':[], 'mse':[], 'mae':[], 'rmse':[], 'r2':[]}

    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_losses = []
        for Xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            with autocast():
                preds = model(Xb)
                loss = criterion(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # VALIDATION using same dataset for simplicity (or implement separate val_split)
        model.eval()
        val_losses = []; all_pred=[]; all_y=[]
        with torch.no_grad():
            for Xv, yv in tqdm(train_loader, desc=f"Val Epoch {epoch+1}", leave=False):
                with autocast():
                    pv = model(Xv)
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
        for k in ['mse','mae','rmse','r2']:
            history[k].append(metrics[k])

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('MSE/val', metrics['mse'], epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | MSE: {metrics['mse']:.6f}")

        # checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.output_dir, 'models', f"{run_id}_best.pth")
            ensure_dir(os.path.dirname(ckpt_path))
            torch.save({'model_state': model.state_dict(), 'config': vars(args)}, ckpt_path)

    # save final model & metrics
    final_path = os.path.join(args.output_dir, 'models', f"{run_id}_final.pth")
    torch.save({'model_state': model.state_dict(), 'config': vars(args)}, final_path)
    save_metrics_csv(history, os.path.join(run_dir, 'history.csv'))
    plot_loss_curves(history, os.path.join(run_dir, 'loss_curve.png'))
    plot_metrics(history, 'mse', os.path.join(run_dir, 'mse_curve.png'))
    writer.close()

    print("✅ Training complete. Run saved to:", run_dir)
    return final_path, run_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs_pipeline")
    parser.add_argument("--model_name", type=str, default="lstm", choices=["lstm"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    train_pipeline(args)

if __name__ == "__main__":
    main()
