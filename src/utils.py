# src/utils.py
import os, math, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ensure_dir(path): 
    os.makedirs(path, exist_ok=True)

def rmse(y, yhat): 
    return math.sqrt(mean_squared_error(y, yhat))

def compute_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rm = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "rmse": rm, "r2": r2}

def save_metrics_csv(metrics_dict, outpath):
    # metrics_dict: {epoch: {metric_name: value, ...}, ...}
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.index.name = 'epoch'
    df.to_csv(outpath)

def plot_loss_curves(history, out_png):
    # history: dict with keys 'train_loss','val_loss','train_mse','val_mse', ...
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
    plt.savefig(out_png, dpi=150); plt.close()

def plot_metrics(history, metric_name, out_png):
    plt.figure(figsize=(8,5))
    plt.plot(history[f'train_{metric_name}'], label=f'train_{metric_name}')
    plt.plot(history[f'val_{metric_name}'], label=f'val_{metric_name}')
    plt.xlabel('epoch'); plt.ylabel(metric_name); plt.legend(); plt.grid(True)
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=150); plt.close()
