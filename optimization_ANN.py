#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_ANN.py
===========
Hyperparameter optimisation for the FCNN architecture using Optuna.

What it does
------------
- Loads a laminar-flame dataset (CSV) via ``load_training_data`` from utils.py.
- Runs a TPE-sampler Optuna study to minimise the mean validation MSE loss
  computed over the **last 1/20th of the training epochs** (e.g. the last 500
  epochs out of 10 000).
- Saves two artefacts:
    * ``best_params.json``  - best hyperparameters + corresponding val loss
    * ``hpo_results.csv``   - full trial history (one row per trial)

Usage
-----
    python optimization_ANN.py \\
        --data_path  /path/to/dataset.csv \\
        --target     flame_speed \\
        --epochs     10000 \\
        --trials     300

Hyperparameters tuned
---------------------
    n_layers     : number of hidden layers
    n_neurons    : neurons per hidden layer (same for all layers)
    batch_size   : mini-batch size (log-uniform over integers)
    lr           : learning rate  (log-uniform)
    weight_decay : L1 regularisation coefficient (log-uniform)

Notes
-----
- Input  dimensionality : 6  (Pressure, Temperature, H2, CO, CO2, H2O mass fractions)
- Output dimensionality : 1  (flame speed *or* density ratio, selected via --target)
- Optimiser             : Adam (fixed)
- Seed                  : 42  (set once at the top via seed_everything)
- Python compatibility  : 3.9+
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Local imports  (adjust relative paths if needed)
# ---------------------------------------------------------------------------
from architectures import FCNN
from utils import load_training_data, TARGET_COLUMNS
from utils_tommaso import seed_everything  # seed_everything lives here

# ---------------------------------------------------------------------------
# Reproducibility  – set once, affects Python / NumPy / PyTorch globally
# ---------------------------------------------------------------------------
SEED = 42
seed_everything(SEED, deterministic=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Optuna HPO for the FCNN architecture."
)

parser.add_argument(
    "--data_path", "-d",
    type=str,
    required=True,
    default='',
    help="Path to the laminar-flame dataset CSV file.",
)
parser.add_argument(
    "--target", "-y",
    type=str,
    choices=["flame_speed", "density_ratio"],
    default="flame_speed",
    help=(
        "Output variable to predict. "
        "Choices: 'flame_speed', 'density_ratio'. "
        "Default: 'flame_speed'."
    ),
)
parser.add_argument(
    "--epochs", "-e",
    type=int,
    default=10_000,
    help="Number of training epochs per trial. Default: 10 000.",
)
parser.add_argument(
    "--trials", "-t",
    type=int,
    default=300,
    help="Number of Optuna trials. Default: 300.",
)
parser.add_argument(
    "--output_dir", "-o",
    type=str,
    default="hpo_output",
    help="Directory where best_params.json and hpo_results.csv are saved. "
         "Default: 'hpo_output'.",
)

args = parser.parse_args()

EPOCHS        = args.epochs
N_TRIALS      = args.trials
TARGET        = args.target
DATA_PATH     = args.data_path
OUTPUT_DIR    = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH         = os.path.join(OUTPUT_DIR, "hpo_results.csv")
BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params.json")
LOG_PATH         = os.path.join(OUTPUT_DIR, "hpo_log.txt")

# ---------------------------------------------------------------------------
# Redirect stdout to log file (mirrors the PBNN HPO convention)
# ---------------------------------------------------------------------------
log_file = open(LOG_PATH, "w")
sys.stdout = log_file

# ---------------------------------------------------------------------------
# Fixed architecture constants
# ---------------------------------------------------------------------------
INPUT_DIM  = 6   # Pressure, Temperature, H2, CO, CO2, H2O mass fractions
OUTPUT_DIM = 1   # flame_speed or density_ratio (single scalar target)

# ---------------------------------------------------------------------------
# Search-space bounds  (tweak here without touching the objective)
# ---------------------------------------------------------------------------
N_LAYERS_LOW,  N_LAYERS_HIGH  = 2,    16
N_NEURONS_LOW, N_NEURONS_HIGH = 16,   128
BATCH_SIZE_LOW, BATCH_SIZE_HIGH = 16, 512   # sampled in log space
LR_LOW,        LR_HIGH        = 1e-7, 1e-1
WD_LOW,        WD_HIGH        = 1e-8, 1e-0

# ---------------------------------------------------------------------------
# Validation-loss window: last 1/20th of epochs
# ---------------------------------------------------------------------------
N_TAIL = max(1, EPOCHS // 20)   # e.g. 500 for 10 000 epochs

# ---------------------------------------------------------------------------
# Load data once (reused across all trials)
# ---------------------------------------------------------------------------
print(f"Loading dataset from: {DATA_PATH}")
print(f"Target variable     : {TARGET}")
print(f"Epochs per trial    : {EPOCHS}")
print(f"Optuna trials       : {N_TRIALS}")
print(f"Val-loss tail length: {N_TAIL} epochs\n")

# load_training_data returns:
#   {"x_train": Tensor, "x_val": Tensor, "y_train": Tensor, "y_val": Tensor}
# FCNN.fit() expects keys "x_train"/"y_train" and "x_test"/"y_test",
# so we rename "x_val"/"y_val" -> "x_test"/"y_test".
_raw = load_training_data(DATA_PATH, target=TARGET, seed=SEED)

data = {
    "x_train": _raw["x_train"],
    "y_train": _raw["y_train"],
    "x_test":  _raw["x_val"],
    "y_test":  _raw["y_val"],
}

print(
    f"Dataset sizes  –  "
    f"train: {data['x_train'].shape[0]}  |  "
    f"val: {data['x_test'].shape[0]}"
)

# ---------------------------------------------------------------------------
# Results accumulator (exported to CSV after the study)
# ---------------------------------------------------------------------------
trial_records = []   # list[dict], one entry per completed trial

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Return the mean validation MSE over the last N_TAIL epochs."""

    # --- suggest hyperparameters -------------------------------------------
    n_layers   = trial.suggest_int("n_layers",   N_LAYERS_LOW,  N_LAYERS_HIGH)
    n_neurons  = trial.suggest_int("n_neurons",  N_NEURONS_LOW, N_NEURONS_HIGH)
    batch_size = trial.suggest_int(
        "batch_size", BATCH_SIZE_LOW, BATCH_SIZE_HIGH, log=True
    )
    lr = trial.suggest_float("lr",           LR_LOW, LR_HIGH, log=True)
    wd = trial.suggest_float("weight_decay", WD_LOW, WD_HIGH, log=True)

    # --- build model --------------------------------------------------------
    model = FCNN(
        input_dim=INPUT_DIM,
        n_layers=n_layers,
        n_neurons=n_neurons,
        output_dim=OUTPUT_DIM,
    )

    # --- train --------------------------------------------------------------
    # verbose=False keeps stdout clean (we only want the Optuna progress bar)
    model.fit(
        data=data,
        epochs=EPOCHS,
        weight_decay=wd,
        learning_rate=lr,
        batch_size=batch_size,
        verbose=False,
        use_gpu=True,
    )

    # --- validation metric --------------------------------------------------
    # FCNN stores per-epoch validation MSE in model.test_mse_loss
    # (cf. architectures.py – the key is test_mse_loss, NOT test_loss_nll)
    val_loss = float(np.mean(np.array(model.test_mse_loss[-N_TAIL:])))

    # --- record for CSV -----------------------------------------------------
    trial_records.append(
        {
            "trial":        trial.number,
            "n_layers":     n_layers,
            "n_neurons":    n_neurons,
            "batch_size":   batch_size,
            "lr":           lr,
            "weight_decay": wd,
            "val_loss":     val_loss,
        }
    )

    return val_loss

# ---------------------------------------------------------------------------
# Run study
# ---------------------------------------------------------------------------
sampler = TPESampler(seed=SEED)
study   = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ---------------------------------------------------------------------------
# Best trial summary
# ---------------------------------------------------------------------------
best          = study.best_trial
best_params   = best.params
best_val_loss = best.value

print("\n" + "=" * 70)
print("HPO complete.")
print(f"  Best validation MSE  : {best_val_loss:.6f}")
print(f"  Best hyperparameters : {best_params}")
print(f"  Val-loss tail length : {N_TAIL} epochs")
print("=" * 70 + "\n")

# ---------------------------------------------------------------------------
# Save best_params.json
# ---------------------------------------------------------------------------
best_params_out = {**best_params, "val_loss": best_val_loss}
with open(BEST_PARAMS_PATH, "w") as fp:
    json.dump(best_params_out, fp, indent=4)
print(f"Best parameters saved  -> {BEST_PARAMS_PATH}")

# ---------------------------------------------------------------------------
# Save hpo_results.csv
# ---------------------------------------------------------------------------
fieldnames = ["trial", "n_layers", "n_neurons", "batch_size",
              "lr", "weight_decay", "val_loss"]

with open(CSV_PATH, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(trial_records)

print(f"Full HPO history saved -> {CSV_PATH}")

log_file.close()