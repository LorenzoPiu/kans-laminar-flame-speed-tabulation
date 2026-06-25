################################################################################
######################## Data loading utilities ################################
################################################################################

"""
Utilities to read the laminar flame dataset and prepare PyTorch training data.

Two entry points:
  read_dataset(path)        -> pandas DataFrame with all 16 columns of the CSV
  load_training_data(path)  -> dict {"x_train", "x_val", "y_train", "y_val"}
                               of torch tensors, ready for network training

X is composed of the six inputs: Pressure, Temperature and the unburnt mass
fractions of H2, CO, CO2, H2O. (O2 and N2 are excluded by default because they
are redundant: they are the air remainder, fully determined by the other four
mass fractions. Set include_o2_n2=True to add them.)

y can be the laminar flame speed only, the density ratio only, or both,
selected via the `target` argument.
"""

import random
import os

import numpy as np
import pandas as pd
import torch

# ------------------------------ column names --------------------------------
INPUT_COLUMNS = [
    "Pressure [Pa]",
    "Temperature [K]",
    "Mass Fraction (H2) Unburnt",
    "Mass Fraction (CO) Unburnt",
    "Mass Fraction (CO2) Unburnt",
    "Mass Fraction (H2O) Unburnt",
]
O2_N2_COLUMNS = [
    "Mass Fraction (O2) Unburnt",
    "Mass Fraction (N2) Unburnt",
]
TARGET_COLUMNS = {
    "flame_speed": ["Laminar Flame Speed S_L [m/s]"],
    "density_ratio": ["Expansion Ratio (Unburnt/Burnt)"],
    "both": ["Laminar Flame Speed S_L [m/s]", "Expansion Ratio (Unburnt/Burnt)"],,
}

# ------------------------------- full reader --------------------------------
def read_dataset(path: str) -> pd.DataFrame:
    """Read the dataset CSV with all its variables into a pandas DataFrame."""
    df = pd.read_csv(path)

    expected = INPUT_COLUMNS + O2_N2_COLUMNS + TARGET_COLUMNS["both"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV at '{path}' is missing expected columns: {missing}")
    return df


# ---------------------------- training-data loader --------------------------
def load_training_data(
    path: str,
    target: str = "both",
    val_split: float = 0.2,
    seed: int = 42,
    include_o2_n2: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Load the CSV and return tensors ready for training.

    Args:
        path: path to the dataset CSV.
        target: "flame_speed", "density_ratio" or "both" (default).
        val_split: validation fraction (default 0.2 -> 80/20 split).
        seed: seed used for the shuffling of the train/val split.
        include_o2_n2: also include the (redundant) O2 and N2 unburnt
            mass fractions in X.
        dtype: dtype of the returned tensors (default float32).
        device: optional device to move the tensors to (e.g. "cuda").

    Returns:
        {"x_train": (N_tr, D), "x_val": (N_val, D),
         "y_train": (N_tr, K), "y_val": (N_val, K)}
        with D = 6 (or 8) inputs and K = 1 or 2 targets.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(
            f"target must be one of {list(TARGET_COLUMNS)}, got '{target}'"
        )
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    df = read_dataset(path)

    x_cols = INPUT_COLUMNS + (O2_N2_COLUMNS if include_o2_n2 else [])
    x = df[x_cols].to_numpy()
    y = df[TARGET_COLUMNS[target]].to_numpy()

    # Shuffled 80/20 (by default) train/val split
    set_seed(seed)
    n = len(df)
    perm = np.random.permutation(n)
    n_val = int(round(n * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    to_tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    return {
        "x_train": to_tensor(x[train_idx]),
        "x_val": to_tensor(x[val_idx]),
        "y_train": to_tensor(y[train_idx]),
        "y_val": to_tensor(y[val_idx]),
    }


################################################################################
############## Random functions reproducibility utilities ######################
################################################################################

def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set random seeds for Python, NumPy, PyTorch, and TensorFlow if available.

    Args:
        seed: The seed value to use.
        deterministic: If True, configures PyTorch for more deterministic behavior.
            This can reduce performance and may raise errors for some operations.

    Returns:
        The seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Helps with deterministic CUDA behavior for some operations.
            # Should be set before CUDA operations are initialized.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    return seed
