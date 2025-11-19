"""
Benchmark models for one-step-ahead Vcorr prediction.

Each model consumes the same feature vector that feeds the Neural ODE
(`V_spme_norm`, `ocv`, `Vcorr`, `SOC`, `I`, `T`) at time step *k* and predicts
`Vcorr(k+1)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import copy
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEFAULT_INPUT_DIM = 6  # [V_spme_norm, ocv, Vcorr, SOC, I, T]


def _build_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    activation: nn.Module = nn.Tanh(),
    init_gain: float = 0.5,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron (MLP).
    
    Uses orthogonal initialization:
    - Hidden layers: gain = 1
    - Output layer: gain = 0 (zero initialization)
    
    Args:
        init_gain: Not used (kept for compatibility). Hidden layers use gain=1, output uses gain=0.
        dropout: Dropout probability (0.0 = no dropout). Applied after activation.
    """
    layers = []
    prev_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        if dropout is not None and dropout > 0.0 and i < len(hidden_dims) - 1:  # Don't add dropout after last hidden layer
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    net = nn.Sequential(*layers)
    
    # Apply orthogonal initialization: hidden layers gain=1, output layer gain=0
    layer_idx = 0
    num_linear_layers = sum(1 for l in net if isinstance(l, nn.Linear))
    for layer in net:
        if isinstance(layer, nn.Linear):
            is_output_layer = (layer_idx == num_linear_layers - 1)
            if is_output_layer:
                # Output layer: gain = 0 (zero initialization)
                nn.init.zeros_(layer.weight)
            else:
                # Hidden layers: gain = 1 (orthogonal)
                nn.init.orthogonal_(layer.weight, gain=1.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layer_idx += 1
    
    return net


class DNNBenchmark(nn.Module):
    """
    Feed-forward network that predicts dY (change in normalized Y) in ResNet style.
    Uses normalized values: V_spme_norm, ocv, Y, I, T are all normalized.
    Prediction: Y(k+1) = Y(k) + dY, where dY is predicted by the network.
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden_dims: Optional[Union[List[int], Tuple[int, ...]]] = None, dropout: float = 0.0):
        super().__init__()
        # Default architecture: [32, 32, 32, 16] (matches saved checkpoint)
        # Can be customized via hidden_dims parameter
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 16]
        # Convert list to tuple for _build_mlp (which expects tuple)
        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)
        self.net = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., input_dim) containing [V_spme_norm, ocv, Y, SOC, I, T]
               where all features are normalized (Y is normalized Vcorr)
        Returns:
            (..., 1) predicted dY (change in normalized Y) for ResNet: Y(k+1) = Y(k) + dY
        """
        return self.net(x)


def _resolve_array(profile: dict, key: str) -> Optional[np.ndarray]:
    if key in profile:
        value = profile[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)
    return None


def _extract_features_and_targets(
    profile: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a profile dict to feature matrix [T, 6], Vcorr vector [T],
    V_spme vector [T], and V_meas vector [T].
    
    Features use normalized values (same as Neural ODE):
    - V_spme_norm, ocv, I, T: normalized
    - Vcorr: normalized (Y = (Vcorr - Y_mean) / Y_std)
    - SOC: already normalized (0~1)
    
    Returns denormalized Vcorr for RMSE calculation only.
    """
    V_spme_norm = _resolve_array(profile, "V_spme_norm")
    if V_spme_norm is None:
        raise KeyError("Profile must contain 'V_spme_norm' (normalized SPME voltage)")
    ocv = _resolve_array(profile, "ocv")
    soc = _resolve_array(profile, "SOC")
    current = _resolve_array(profile, "I")
    temperature = _resolve_array(profile, "T")

    v_spme = _resolve_array(profile, "V_spme")
    v_meas = _resolve_array(profile, "V_meas")

    # Get normalized Y for features (same as Neural ODE)
    Y_norm = _resolve_array(profile, "Y")
    if Y_norm is None:
        # Try to compute from Vcorr or V_meas/V_spme
        vcorr_raw = _resolve_array(profile, "Vcorr")
        if vcorr_raw is None:
            if v_meas is not None and v_spme is not None:
                vcorr_raw = v_meas - v_spme
            else:
                raise KeyError(
                    "Profile must contain 'Y' (normalized), 'Vcorr', "
                    "or both 'V_meas' and 'V_spme'."
                )
        
        # Normalize Vcorr if Y_mean and Y_std are available
        if "Y_mean" in profile and "Y_std" in profile:
            Y_mean = float(profile["Y_mean"])
            Y_std = float(profile["Y_std"])
            Y_norm = (vcorr_raw - Y_mean) / Y_std
        else:
            raise KeyError(
                "Profile must contain 'Y' (normalized) or 'Vcorr' with 'Y_mean' and 'Y_std'."
            )
    
    # Get Y_mean and Y_std for denormalization (for RMSE calculation)
    if "Y_mean" in profile and "Y_std" in profile:
        Y_mean = float(profile["Y_mean"])
        Y_std = float(profile["Y_std"])
        vcorr_denorm = Y_norm * Y_std + Y_mean  # Denormalized for RMSE
    else:
        raise KeyError("Profile must contain 'Y_mean' and 'Y_std' for denormalization.")

    if any(arr is None for arr in (V_spme_norm, ocv, soc, current, temperature, v_spme, v_meas)):
        missing = [
            name
            for name, arr in zip(
                ["V_spme_norm", "ocv", "SOC", "I", "T", "V_spme", "V_meas"],
                (V_spme_norm, ocv, soc, current, temperature, v_spme, v_meas),
            )
            if arr is None
        ]
        raise KeyError(f"Missing required arrays: {missing}")

    min_len = min(
        len(V_spme_norm), len(ocv), len(Y_norm), len(soc), len(current), len(temperature), len(v_spme), len(v_meas)
    )
    if min_len < 2:
        raise ValueError("Each profile must contain at least two time steps.")

    # Features use normalized Y (same as Neural ODE)
    features = np.stack(
        [
            V_spme_norm[:min_len],  # V_spme_norm (already normalized)
            ocv[:min_len],
            Y_norm[:min_len],  # Normalized Y (not denormalized Vcorr)
            soc[:min_len],
            current[:min_len],
            temperature[:min_len],
        ],
        axis=-1,
    ).astype(np.float32)
    vcorr_denorm = vcorr_denorm[:min_len].astype(np.float32)
    v_spme = v_spme[:min_len].astype(np.float32)
    v_meas = v_meas[:min_len].astype(np.float32)
    return features, vcorr_denorm, v_spme, v_meas


class SingleStepDataset(Dataset):
    """
    Dataset for one-step predictors such as DNNBenchmark.
    
    Uses normalized space (same as Neural ODE):
    - Input features: normalized (V_spme_norm, ocv, Y, SOC, I, T)
    - Targets: normalized Y (Y(k+1))
    """

    def __init__(self, profiles: Sequence[dict]):
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        v_spme_next: List[np.ndarray] = []
        v_meas_next: List[np.ndarray] = []
        
        for profile_idx, profile in enumerate(profiles):
            features, vcorr_denorm, v_spme, v_meas = _extract_features_and_targets(profile)
            
            # Features contain normalized Y at position 2
            # Targets should be normalized Y(k+1)
            # features: [V_spme_norm, ocv, Y, SOC, I, T] where Y is normalized
            # targets: Y(k+1) normalized
            
            xs.append(features[:-1])
            ys.append(features[1:, 2:3])  # Normalized Y(k+1) as target
            v_spme_next.append(v_spme[1:, None])
            v_meas_next.append(v_meas[1:, None])

        if not xs:
            raise ValueError("No samples generated – check input profiles.")

        self.inputs = torch.from_numpy(np.concatenate(xs, axis=0))
        self.targets = torch.from_numpy(np.concatenate(ys, axis=0))
        self.v_spme_next = torch.from_numpy(np.concatenate(v_spme_next, axis=0))
        self.v_meas_next = torch.from_numpy(np.concatenate(v_meas_next, axis=0))
        
        # Store Y_std_avg for denormalization (for RMSE display)
        Y_std_list: List[float] = []
        for profile in profiles:
            if "Y_std" in profile:
                Y_std_list.append(float(profile["Y_std"]))
        self.Y_std_avg = np.mean(Y_std_list) if Y_std_list else 0.004  # Default Y_std

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self.v_spme_next[idx], self.v_meas_next[idx]


class LSTMBenchmark(nn.Module):
    """
    LSTM-based one-step predictor with a comparable capacity to the FNN head.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: shape (batch, seq_len, input_dim) or (seq_len, input_dim)
        Returns:
            y: shape (batch, 1) predicted Vcorr(k+1) from the last time step
            (hn, cn): LSTM hidden states for rollouts
        """
        if x.dim() == 2:  # (seq_len, input_dim)
            x = x.unsqueeze(0)
        output, (hn, cn) = self.lstm(x, hx)
        y = self.head(output[:, -1, :])
        return y, (hn, cn)


class GRUBenchmark(nn.Module):
    """
    GRU-based predictor with capacity similar to the LSTM benchmark.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape (batch, seq_len, input_dim) or (seq_len, input_dim)
        Returns:
            y: shape (batch, 1) predicted Vcorr(k+1)
            hn: GRU hidden state for continued rollouts
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        output, hn = self.gru(x, hx)
        y = self.head(output[:, -1, :])
        return y, hn


class SequenceDataset(Dataset):
    """
    Dataset that returns sliding windows for sequence models (LSTM/GRU/Transformer).
    """

    def __init__(self, profiles: Sequence[dict], seq_len: int):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        sequences: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        for profile in profiles:
            features, vcorr, _, _ = _extract_features_and_targets(profile)
            if len(features) <= seq_len:
                continue
            for end in range(seq_len, len(features)):
                start = end - seq_len
                sequences.append(features[start:end])
                targets.append(vcorr[end])

        if not sequences:
            raise ValueError("No sequences generated – reduce seq_len or check input profiles.")

        self.sequences = torch.from_numpy(np.stack(sequences).astype(np.float32))
        self.targets = torch.from_numpy(np.array(targets, dtype=np.float32)[:, None])

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


@dataclass
class TransformerConfig:
    input_dim: int = DEFAULT_INPUT_DIM
    model_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class TransformerBenchmark(nn.Module):
    """
    Prepared scaffold for a transformer-based predictor.

    Currently wires an encoder stack followed by a linear head, but the exact
    training/inference loop remains to be implemented.
    """

    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, input_dim)
        Returns:
            (batch, 1) predicted Vcorr(k+1) using the final token representation.
        """
        if x.dim() != 3:
            raise ValueError("TransformerBenchmark expects input of shape (batch, seq_len, input_dim)")
        emb = self.input_proj(x)
        encoded = self.encoder(emb)
        return self.head(encoded[:, -1, :])


def build_dnn_dataset(dict_list: Sequence[dict]) -> SingleStepDataset:
    """
    Convenience wrapper that mirrors the neural-ODE data interface.
    """
    return SingleStepDataset(dict_list)


def build_sequence_dataset(dict_list: Sequence[dict], seq_len: int) -> SequenceDataset:
    """
    Convenience wrapper for LSTM/GRU/Transformer benchmarks.
    """
    return SequenceDataset(dict_list, seq_len)


def make_benchmark_dataset(
    dict_list: Sequence[dict],
    model_kind: str,
    seq_len: Optional[int] = None,
):
    """
    Create the appropriate dataset object for the requested benchmark model.
    """
    model_kind = model_kind.lower()
    if model_kind == "dnn":
        return build_dnn_dataset(dict_list)
    if model_kind in {"lstm", "gru", "transformer"}:
        if seq_len is None or seq_len < 1:
            raise ValueError("seq_len must be provided and >= 1 for sequence models.")
        return build_sequence_dataset(dict_list, seq_len)
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def make_benchmark_dataloader(
    dict_list: Sequence[dict],
    model_kind: str,
    *,
    seq_len: Optional[int] = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
) -> Tuple[Dataset, DataLoader]:
    """
    Build a dataset/dataloader pair with consistent defaults across benchmarks.

    Args:
        dict_list: list of profile dicts.
        model_kind: one of {"dnn", "lstm", "gru", "transformer"}.
        seq_len: window length for sequence models (ignored for DNN).
        batch_size: per-step batch size.
        shuffle: shuffle dataset each epoch.
        drop_last: drop remainder batch (keeps tensor shapes stable).
        num_workers: DataLoader worker count (defaults to up to 4 CPU cores).
        pin_memory: enable pinned host memory for faster GPU transfers.
        persistent_workers: keep workers alive between epochs.
    """
    dataset = make_benchmark_dataset(dict_list, model_kind, seq_len)
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return dataset, loader


@dataclass
class BenchmarkTrainingSummary:
    model_kind: str
    model: nn.Module
    architecture: str
    state_dict_path: Path
    num_epochs: int
    lr: float
    batch_size: int
    num_workers: int
    best_epoch: int
    best_rmse_mV: float
    history: Dict[str, List[float]]


def _default_artifact_path(model_kind: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir / f"{model_kind}_benchmark_{timestamp}.pth"


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_pretrained_state(model: nn.Module, pretrained_path: Optional[Union[str, Path]]) -> bool:
    if pretrained_path is None:
        return False
    path = Path(pretrained_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained model not found at: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    return True


def train_dnn_benchmark(
    train_dict_list: Sequence[dict],
    num_epochs: int,
    lr: float,
    device: Union[str, torch.device],
    training_batch_size: Optional[int] = None,
    num_workers: int = 0,
    *,
    val_dict_list: Optional[Sequence[dict]] = None,
    pretrained_model_path: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    clip_grad_norm: float = 50.0,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-6,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    l2_reg: float = 0.0,
    verbose: bool = True,
    hidden_dims: Optional[Union[List[int], Tuple[int, ...]]] = None,
    dropout: float = 0.0,
) -> BenchmarkTrainingSummary:
    """
    Train the DNN benchmark model using profile-wise training (similar to Neural ODE).
    
    Args:
        train_dict_list: List of profile dicts for training
        val_dict_list: Optional list of profile dicts for validation
        training_batch_size: Number of profiles to use per training step (None = use all)
        l2_reg: L2 regularization coefficient. If > 0, adds L2 penalty to loss (default: 0.0)
        hidden_dims: List or Tuple of hidden layer dimensions (e.g., [32, 32, 32, 16])
                    None = use default [32, 32, 32, 16]
                    Examples:
                      [64, 64, 64, 32]  # Medium capacity
                      [128, 128, 64, 32]  # High capacity
                      [32, 32, 32, 16]  # Default (original)
        dropout: Dropout probability for regularization (0.0 = no dropout)
                Recommended: 0.1-0.3 for better generalization
    """
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = _resolve_device(device)

    total_profiles = len(train_dict_list)
    training_batch_size = training_batch_size if training_batch_size is not None else total_profiles
    training_batch_size = min(training_batch_size, total_profiles)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[DNN] Profile-wise Training: {total_profiles} total profiles")
        print(f"[DNN] Training batch size: {training_batch_size} profiles per epoch")
        print(f"{'='*60}")
        sys.stdout.flush()
    
    # Extract features and targets for all training profiles
    train_features_list = []
    train_targets_list = []
    train_valid_lengths = []
    Y_std_list = []
    
    for profile in train_dict_list:
        features, vcorr_denorm, v_spme, v_meas = _extract_features_and_targets(profile)
        # Get Y_mean and Y_std for this profile
        if "Y_mean" in profile and "Y_std" in profile:
            Y_mean = float(profile["Y_mean"])
            Y_std = float(profile["Y_std"])
            Y_std_list.append(Y_std)
        else:
            raise KeyError("Profile must contain 'Y_mean' and 'Y_std'.")
        
        # Features contain normalized Y at position 2
        # Target is normalized Y (features[1:, 2] contains Y(k+1))
        target_y_norm = features[1:, 2]  # Normalized Y(k+1), shape: [T-1]
        
        train_features_list.append(features)  # Shape: [T, 6]
        train_targets_list.append(target_y_norm)  # Shape: [T-1]
        train_valid_lengths.append(len(features))
    
    Y_std_avg = np.mean(Y_std_list)
    
    # Find max length for padding
    max_train_length = max(train_valid_lengths) if train_valid_lengths else 0
    
    # Extract features and targets for validation profiles (if provided)
    val_features_list = []
    val_targets_list = []
    val_valid_lengths = []
    
    if val_dict_list is not None and len(val_dict_list) > 0:
        for profile in val_dict_list:
            features, vcorr_denorm, v_spme, v_meas = _extract_features_and_targets(profile)
            if "Y_mean" in profile and "Y_std" in profile:
                Y_mean = float(profile["Y_mean"])
                Y_std = float(profile["Y_std"])
            else:
                raise KeyError("Profile must contain 'Y_mean' and 'Y_std'.")
            
            target_y_norm = features[1:, 2]  # Shape: [T-1]
            
            val_features_list.append(features)  # Shape: [T, 6]
            val_targets_list.append(target_y_norm)  # Shape: [T-1]
            val_valid_lengths.append(len(features))
        
        max_val_length = max(val_valid_lengths) if val_valid_lengths else 0
    else:
        max_val_length = 0

    if verbose:
        print(f"[DNN] Initializing model...")
        sys.stdout.flush()
    model = DNNBenchmark(input_dim=DEFAULT_INPUT_DIM, hidden_dims=hidden_dims, dropout=dropout).to(device)
    _load_pretrained_state(model, pretrained_model_path)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[DNN] Model architecture: {hidden_dims if hidden_dims else [32, 32, 32, 16]} (hidden_dims)")
        print(f"[DNN] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        sys.stdout.flush()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
        threshold=0.005,       # 0.005 mV
        threshold_mode="abs",
    )
    criterion = torch.nn.MSELoss(reduction='sum')  # Sum reduction for profile-wise loss
    
    if verbose:
        print(f"[DNN] Starting training for {num_epochs} epochs...")
        print(f"[DNN] Device: {device}, Learning rate: {lr:.6f}")
        print("=" * 80)
        sys.stdout.flush()

    best_rmse = float("inf")  # Normalized RMSE (same as Neural ODE)
    best_epoch = -1
    best_state_dict = None
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_rmse_mV": [],
        "val_loss": [],
        "val_rmse_mV": [],
        "lr": [],
        "grad_before": [],
        "grad_after": [],
    }
    rmse_window = deque(maxlen=early_stop_window)
    use_validation = len(val_features_list) > 0

    for epoch in range(1, num_epochs + 1):
        # ===== Training Phase =====
        model.train()
        optimizer.zero_grad()
        
        # Sample random batch of profiles if training_batch_size < total_profiles
        if training_batch_size < total_profiles:
            selected_indices = np.random.choice(total_profiles, training_batch_size, replace=False)
            selected_features = [train_features_list[i] for i in selected_indices]
            selected_targets = [train_targets_list[i] for i in selected_indices]
            selected_valid_lengths = [train_valid_lengths[i] for i in selected_indices]
            batch_size = training_batch_size
            max_length = max(selected_valid_lengths)
        else:
            selected_features = train_features_list
            selected_targets = train_targets_list
            selected_valid_lengths = train_valid_lengths
            batch_size = total_profiles
            max_length = max_train_length
        
        if verbose and epoch == 1:
            print(f"[DNN] Starting epoch {epoch}/{num_epochs}...")
            sys.stdout.flush()
        
        # Prepare batch tensors with padding
        features_batch = torch.zeros(batch_size, max_length, 6, dtype=torch.float32, device=device)
        targets_batch = torch.zeros(batch_size, max_length - 1, dtype=torch.float32, device=device)
        valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        
        for i, (features, targets, valid_len) in enumerate(zip(selected_features, selected_targets, selected_valid_lengths)):
            features_tensor = torch.from_numpy(features).to(device, dtype=torch.float32)
            targets_tensor = torch.from_numpy(targets).to(device, dtype=torch.float32)
            
            features_batch[i, :valid_len] = features_tensor
            targets_batch[i, :valid_len-1] = targets_tensor  # targets is [T-1]
            valid_mask[i, :valid_len] = True
        
        # Perform batch autoregressive rollout (with gradients)
        pred_y_norm_batch = _dnn_autoregressive_rollout_train_batch(
            model, features_batch, valid_mask
        )  # Shape: [batch_size, max_length]
        
        # Calculate loss on entire sequences (exclude first time step as it's the initial condition)
        # pred_y_norm_batch[:, 1:] corresponds to k=1,2,...,T-1 (predictions)
        # targets_batch corresponds to k=1,2,...,T-1 (targets)
        pred_seq_batch = pred_y_norm_batch[:, 1:]  # Shape: [batch_size, max_length-1]
        
        # Create mask for valid target positions (excluding padding)
        valid_target_mask = valid_mask[:, 1:]  # Shape: [batch_size, max_length-1]
        
        # Calculate loss only on valid positions (vectorized for speed)
        valid_target_mask_float = valid_target_mask.float()
        squared_diff = (pred_seq_batch - targets_batch) ** 2
        loss_V = (squared_diff * valid_target_mask_float).sum()
        num_valid = valid_target_mask.sum().item()
        
        # Check for NaN or Inf in loss
        if torch.isnan(loss_V) or torch.isinf(loss_V):
            raise ValueError(f"NaN or Inf detected in loss_V. pred_seq range: [{pred_seq_batch.min().item():.2f}, {pred_seq_batch.max().item():.2f}], "
                           f"targets range: [{targets_batch.min().item():.2f}, {targets_batch.max().item():.2f}]")
        
        # Average loss over all valid points
        avg_loss = loss_V / num_valid if num_valid > 0 else loss_V
        
        # Add L2 regularization if specified
        if l2_reg > 0.0:
            l2_loss = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    l2_loss += param.norm(2) ** 2
            avg_loss = avg_loss + l2_reg * l2_loss
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient norm
        grad_before = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq = param.grad.data.norm(2).item() ** 2
                if np.isnan(grad_norm_sq) or np.isinf(grad_norm_sq):
                    # Replace NaN/Inf with a large finite value for gradient clipping
                    grad_norm_sq = 1e10
                grad_before += grad_norm_sq
        grad_norm_before = grad_before ** 0.5
        if np.isnan(grad_norm_before) or np.isinf(grad_norm_before):
            grad_norm_before = 1e10

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        grad_after = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq = param.grad.data.norm(2).item() ** 2
                if np.isnan(grad_norm_sq) or np.isinf(grad_norm_sq):
                    grad_norm_sq = 1e10
                grad_after += grad_norm_sq
        grad_norm_after = grad_after ** 0.5
        if np.isnan(grad_norm_after) or np.isinf(grad_norm_after):
            grad_norm_after = 1e10

        optimizer.step()

        train_epoch_loss = avg_loss.item()
        train_rmse_norm = train_epoch_loss ** 0.5
        train_rmse_mV = train_rmse_norm * Y_std_avg * 1000  # Convert to mV for display
        
        # ===== Validation Phase =====
        val_epoch_loss = None
        val_epoch_rmse_mV = None
        if use_validation:
            model.eval()
            
            with torch.no_grad():
                # Batch validation (same as training, autoregressive rollout)
                val_batch_size = len(val_features_list)
                val_max_length = max_val_length
                
                # Prepare batch tensors with padding
                val_features_batch = torch.zeros(val_batch_size, val_max_length, 6, dtype=torch.float32, device=device)
                val_targets_batch = torch.zeros(val_batch_size, val_max_length - 1, dtype=torch.float32, device=device)
                val_valid_mask = torch.zeros(val_batch_size, val_max_length, dtype=torch.bool, device=device)
                
                for i, (features, targets, valid_len) in enumerate(zip(val_features_list, val_targets_list, val_valid_lengths)):
                    features_tensor = torch.from_numpy(features).to(device, dtype=torch.float32)
                    targets_tensor = torch.from_numpy(targets).to(device, dtype=torch.float32)
                    
                    val_features_batch[i, :valid_len] = features_tensor
                    val_targets_batch[i, :valid_len-1] = targets_tensor
                    val_valid_mask[i, :valid_len] = True
                
                # Perform batch autoregressive rollout
                val_pred_y_norm_batch = _dnn_autoregressive_rollout_train_batch(
                    model, val_features_batch, val_valid_mask
                )  # Shape: [batch_size, max_length]
                
                # Calculate loss on entire sequences
                val_pred_seq_batch = val_pred_y_norm_batch[:, 1:]  # Shape: [batch_size, max_length-1]
                val_valid_target_mask = val_valid_mask[:, 1:]  # Shape: [batch_size, max_length-1]
                
                # Calculate loss only on valid positions (vectorized for speed)
                val_valid_target_mask_float = val_valid_target_mask.float()
                val_squared_diff = (val_pred_seq_batch - val_targets_batch) ** 2
                val_loss_V = (val_squared_diff * val_valid_target_mask_float).sum()
                val_num_valid = val_valid_target_mask.sum().item()
                
                # Average loss over all valid points
                val_epoch_loss = (val_loss_V / val_num_valid).item() if val_num_valid > 0 else val_loss_V.item()
                val_rmse_norm = val_epoch_loss ** 0.5
                val_epoch_rmse_mV = val_rmse_norm * Y_std_avg * 1000  # Convert to mV for display
        
        # Use validation mV RMSE for scheduler if available, else training mV RMSE
        monitor_mV = val_epoch_rmse_mV if (use_validation and val_epoch_rmse_mV is not None) else train_rmse_mV
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(monitor_mV)
        new_lr = optimizer.param_groups[0]["lr"]
        
        # Check if LR was reduced by scheduler
        if new_lr < current_lr and verbose:
            print(f"[DNN] ⚠️  Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_epoch_loss)
        history["train_rmse_mV"].append(train_rmse_mV)  # Denormalized RMSE in mV (same as Neural ODE)
        history["val_loss"].append(val_epoch_loss if val_epoch_loss is not None else float("nan"))
        history["val_rmse_mV"].append(val_epoch_rmse_mV if val_epoch_rmse_mV is not None else float("nan"))  # Denormalized RMSE in mV (same as Neural ODE)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grad_before"].append(grad_norm_before)
        history["grad_after"].append(grad_norm_after)

        # Print every epoch if verbose
        if verbose:
            if use_validation:
                print(
                    f"[DNN] epoch {epoch}/{num_epochs}, "
                    f"train_loss={train_epoch_loss:.3e}, train_RMSE={train_rmse_mV:.2f} mV, "
                    f"val_loss={val_epoch_loss:.3e}, val_RMSE={val_epoch_rmse_mV:.2f} mV, "
                    f"LR={optimizer.param_groups[0]['lr']:.3e}, "
                    f"Grad={grad_norm_before:.2f}->{grad_norm_after:.2f}"
                )
            else:
                print(
                    f"[DNN] epoch {epoch}/{num_epochs}, "
                    f"loss={train_epoch_loss:.3e}, "
                    f"RMSE={train_rmse_mV:.2f} mV, "
                    f"LR={optimizer.param_groups[0]['lr']:.3e}, "
                    f"Grad={grad_norm_before:.2f}->{grad_norm_after:.2f}"
                )

        # Use normalized RMSE for monitoring (same as Neural ODE)
        # All calculations in normalized space, only display is denormalized
        
        # Best model selection: use validation if available (to prevent overfitting)
        if use_validation and val_epoch_rmse_mV is not None:
            best_monitor_rmse_norm = val_rmse_norm  # Normalized RMSE
            best_monitor_rmse_mV = val_epoch_rmse_mV  # Denormalized for display
            best_metric_name = "val_RMSE"
        else:
            best_monitor_rmse_norm = train_rmse_norm  # Normalized RMSE
            best_monitor_rmse_mV = train_rmse_mV  # Denormalized for display
            best_metric_name = "RMSE"
        
        # Early stopping: use training RMSE (more stable, reflects actual learning progress)
        early_stop_monitor_rmse_norm = train_rmse_norm  # Normalized RMSE
        early_stop_monitor_rmse_mV = train_rmse_mV  # Denormalized for display
        
        # Update best model based on validation (if available) or training
        rmse_window.append(early_stop_monitor_rmse_mV)  # Early stopping uses training RMSE
        if best_monitor_rmse_norm < best_rmse:
            best_rmse = best_monitor_rmse_norm  # Store normalized RMSE
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"[DNN] ✅ Best {best_metric_name} improved to {best_monitor_rmse_mV:.2f} mV at epoch {epoch}")

        # Early stopping based on training RMSE (more stable)
        if (
            rmse_window.maxlen == len(rmse_window)
            and (max(rmse_window) - min(rmse_window)) <= early_stop_delta_mV
        ):
            if verbose:
                print(f"[DNN] Early stopping triggered (Δtrain_RMSE <= {early_stop_delta_mV:.3f} mV over {early_stop_window} epochs)")
            break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)

    # Use Neural ODE naming convention: best_model_batch_rmse{rmse}mV.pth
    if save_path is None:
        best_rmse_mV = best_rmse * Y_std_avg * 1000
        save_path = f"best_model_batch_rmse{best_rmse_mV:.2f}mV.pth"
    save_path = Path(save_path)
    
    payload = {
        "model_kind": "dnn",
        "state_dict": best_state_dict,
        "model_kwargs": {"input_dim": DEFAULT_INPUT_DIM, "hidden_dims": list(hidden_dims) if isinstance(hidden_dims, tuple) else hidden_dims, "dropout": dropout},
        "architecture": repr(model),
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": training_batch_size,
        "num_workers": num_workers,
        "best_epoch": best_epoch,
        "best_rmse_mV": best_rmse * Y_std_avg * 1000,  # Denormalize for storage (same as Neural ODE)
        "history": history,
    }
    torch.save(payload, save_path)

    summary = BenchmarkTrainingSummary(
        model_kind="dnn",
        model=model.cpu(),
        architecture=repr(model),
        state_dict_path=save_path,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=training_batch_size,
        num_workers=num_workers,
        best_epoch=best_epoch,
        best_rmse_mV=best_rmse * Y_std_avg * 1000,  # Denormalize for storage (same as Neural ODE)
        history=history,
    )

    if verbose:
        print(f"[DNN] Best RMSE = {summary.best_rmse_mV:.2f} mV at epoch {summary.best_epoch}")
        print(f"[DNN] Architecture:")
        print(summary.architecture)
        print(f"[DNN] Best model saved to: {save_path}")
        print(f"{'='*60}\n")

    return summary


def load_dnn_benchmark_model(
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    hidden_dims: Optional[Union[List[int], Tuple[int, ...]]] = None,
    dropout: Optional[float] = None,
) -> Tuple[DNNBenchmark, Dict[str, Union[int, float, str, Dict[str, List[float]]]]]:
    """
    Load a serialized DNN benchmark checkpoint and return the model plus metadata.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
        hidden_dims: Optional hidden layer dimensions. If None, tries to load from checkpoint.
                    If checkpoint doesn't have it, uses default (32, 32, 32, 16).
    """
    device = _resolve_device(device)
    
    # Try to load model_kwargs from checkpoint
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_kwargs" in payload:
        model_kwargs = payload["model_kwargs"]
        # Use hidden_dims from checkpoint if not provided
        if hidden_dims is None and "hidden_dims" in model_kwargs:
            hidden_dims = model_kwargs["hidden_dims"]
            # Convert to list if it's a tuple (for consistency)
            if isinstance(hidden_dims, tuple):
                hidden_dims = list(hidden_dims)
        # Use dropout from checkpoint if not provided
        if dropout is None and "dropout" in model_kwargs:
            dropout = model_kwargs["dropout"]
        input_dim = model_kwargs.get("input_dim", DEFAULT_INPUT_DIM)
    else:
        input_dim = DEFAULT_INPUT_DIM
    
    # Set default dropout if None
    if dropout is None:
        dropout = 0.0
    
    model = DNNBenchmark(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.eval()

    metadata: Dict[str, Union[int, float, str, Dict[str, List[float]]]] = {}
    if isinstance(payload, dict):
        metadata = {
            "model_kind": payload.get("model_kind", "dnn"),
            "num_epochs": payload.get("num_epochs"),
            "lr": payload.get("lr"),
            "batch_size": payload.get("batch_size"),
            "num_workers": payload.get("num_workers"),
            "best_epoch": payload.get("best_epoch"),
            "best_rmse_mV": payload.get("best_rmse_mV"),
            "history": payload.get("history"),
            "saved_at": str(Path(model_path)),
        }
    return model, metadata


def _resolve_profile_name(profile: dict, fallback: str) -> str:
    for key in ("name", "profile_name", "file_name", "id", "profile_id"):
        if key in profile and profile[key] is not None:
            return str(profile[key])
    return fallback


def _dnn_autoregressive_rollout_train_batch(
    model: DNNBenchmark,
    features_batch: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Perform autoregressive rollout for training on a batch of profiles (gradient enabled).
    
    Args:
        model: DNNBenchmark model
        features_batch: torch.Tensor shape [batch_size, T_max, 6] containing normalized values [V_spme_norm, ocv, Y, SOC, I, T]
        valid_mask: torch.Tensor shape [batch_size, T_max] of bool indicating valid (non-padded) time steps
    Returns:
        pred_y_norm: torch.Tensor shape [batch_size, T_max] with predicted normalized Y (requires_grad=True)
    """
    batch_size, T_max, _ = features_batch.shape
    device = features_batch.device
    
    # Initialize predictions: [batch_size, T_max]
    # Use list to avoid in-place assignment issues with gradients
    pred_y_norm_list = []
    
    # Initial condition: normalized Y at k=0 (only for valid profiles)
    # Detach initial condition to prevent gradient flow from ground truth
    y_k = features_batch[:, 0, 2].detach()  # Y(k=0) normalized, no gradient
    pred_y_norm_list.append(y_k)
    
    # Autoregressive rollout for each time step (ResNet style: Y(k+1) = Y(k) + dY)
    for k in range(T_max - 1):
        # Build input batch using current y_k (which has gradient if k > 0)
        input_batch = torch.stack([
            features_batch[:, k, 0],  # V_spme_norm (normalized)
            features_batch[:, k, 1],  # ocv (normalized)
            y_k,                       # autoregressive Y(k) (normalized) - has gradient if k > 0
            features_batch[:, k, 3],  # SOC (normalized)
            features_batch[:, k, 4],  # current (normalized)
            features_batch[:, k, 5],  # temperature (normalized)
        ], dim=1)  # [batch_size, 6]
        
        # Batch prediction: dY (change in normalized Y) in ResNet style
        pred_dy_norm = model(input_batch).squeeze(-1)  # [batch_size, 1] -> [batch_size]
        # ResNet: Y(k+1) = Y(k) + dY
        # Create new tensor (not in-place) to preserve gradient flow
        y_k = y_k + pred_dy_norm  # New tensor with gradient
        pred_y_norm_list.append(y_k)
    
    # Stack all predictions into a single tensor
    pred_y_norm = torch.stack(pred_y_norm_list, dim=1)  # [batch_size, T_max]
    return pred_y_norm


def _dnn_autoregressive_rollout_inference_batch(
    model: DNNBenchmark,
    features_batch: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Autoregressive rollout for inference on a batch of profiles (no gradients).
    """
    batch_size, T_max, _ = features_batch.shape
    device = features_batch.device
    pred_y_norm = torch.zeros(batch_size, T_max, dtype=torch.float32, device=device)
    pred_y_norm[:, 0] = features_batch[:, 0, 2]

    with torch.no_grad():
        for k in range(T_max - 1):
            input_batch = torch.stack(
                [
                    features_batch[:, k, 0],  # V_spme_norm (normalized)
                    features_batch[:, k, 1],  # ocv (normalized)
                    pred_y_norm[:, k],        # autoregressive Y(k) (normalized)
                    features_batch[:, k, 3],  # SOC (normalized)
                    features_batch[:, k, 4],  # current (normalized)
                    features_batch[:, k, 5],  # temperature (normalized)
                ],
                dim=1,
            )
            pred_dy_norm = model(input_batch).squeeze(-1)
            pred_y_norm[:, k + 1] = pred_y_norm[:, k] + pred_dy_norm

    return pred_y_norm


def _dnn_autoregressive_rollout_train(
    model: DNNBenchmark,
    features: torch.Tensor,
    target_y_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform an autoregressive rollout for training (gradient enabled) - single profile.
    Kept for backward compatibility but not used in batch training.
    
    Args:
        model: DNNBenchmark model
        features: torch.Tensor shape [T, 6] containing normalized values [V_spme_norm, ocv, Y, SOC, I, T]
        target_y_norm: torch.Tensor shape [T] containing target normalized Y (ground truth)
    Returns:
        pred_y_norm: torch.Tensor shape [T] with predicted normalized Y (requires_grad=True)
        target_y_norm: torch.Tensor shape [T] with target normalized Y (same as input)
    """
    seq_len = features.shape[0]
    pred_y_norm = torch.zeros(seq_len, dtype=torch.float32, device=features.device, requires_grad=False)
    pred_y_norm[0] = features[0, 2]  # Initial condition: normalized Y
    
    for k in range(seq_len - 1):
        input_vec = torch.stack(
            [
                features[k, 0],  # V_spme_norm (normalized)
                features[k, 1],  # ocv (normalized)
                pred_y_norm[k],  # autoregressive Y(k) (normalized)
                features[k, 3],  # SOC (normalized)
                features[k, 4],  # current (normalized)
                features[k, 5],  # temperature (normalized)
            ]
        ).unsqueeze(0)
        # ResNet style: predict dY (change in normalized Y)
        pred_dy_norm = model(input_vec).squeeze(0).squeeze(0)
        # ResNet: Y(k+1) = Y(k) + dY
        pred_y_norm[k + 1] = pred_y_norm[k] + pred_dy_norm
    
    return pred_y_norm, target_y_norm


def _dnn_autoregressive_rollout(
    model: DNNBenchmark,
    features: np.ndarray,
    device: torch.device,
    Y_mean: float,
    Y_std: float,
    debug: bool = False,
) -> np.ndarray:
    """
    Perform an autoregressive rollout over a single profile (inference, no gradient).

    Args:
        model: trained DNNBenchmark
        features: np.ndarray shape [T, 6] as returned by _extract_features_and_targets
                 Contains normalized values: [V_spme_norm, ocv, Y, SOC, I, T]
        device: torch device
        Y_mean: mean for denormalization
        Y_std: std for denormalization
        debug: if True, print debug info for first 3 steps
    Returns:
        np.ndarray shape [T] with predicted Vcorr in physical units (denormalized, mV)
    """
    seq_len = features.shape[0]
    static_features = torch.from_numpy(features).to(device, dtype=torch.float32)
    pred_y_norm = torch.zeros(seq_len, dtype=torch.float32, device=device)
    pred_y_norm[0] = static_features[0, 2]  # Initial condition: normalized Y

    with torch.no_grad():
        for k in range(seq_len - 1):
            input_vec = torch.stack(
                [
                    static_features[k, 0],  # V_spme_norm (normalized)
                    static_features[k, 1],  # ocv (normalized)
                    pred_y_norm[k],          # autoregressive Y(k) (normalized)
                    static_features[k, 3],  # SOC (normalized)
                    static_features[k, 4],  # current (normalized)
                    static_features[k, 5],  # temperature (normalized)
                ]
            ).unsqueeze(0)
            # ResNet style: predict dY (change in normalized Y)
            pred_dy_norm = model(input_vec).squeeze(0).squeeze(0)
            # Debug: check for NaN or Inf
            if torch.isnan(pred_dy_norm) or torch.isinf(pred_dy_norm):
                raise ValueError(f"NaN or Inf detected in pred_dy_norm at k={k}")
            
            # ResNet: Y(k+1) = Y(k) + dY
            pred_y_norm[k + 1] = pred_y_norm[k] + pred_dy_norm
            
            # Debug: print first few steps if requested (only for first profile)
            if debug and k < 3:
                actual_y_norm_k = static_features[k, 2].item()
                actual_y_norm_kp1 = static_features[k+1, 2].item() if k+1 < seq_len else None
                pred_y_norm_k = pred_y_norm[k].item()
                pred_y_norm_kp1 = pred_y_norm[k + 1].item()
                
                # Denormalize for display
                pred_vcorr_k = pred_y_norm_k * Y_std + Y_mean
                actual_vcorr_k = actual_y_norm_k * Y_std + Y_mean
                pred_vcorr_kp1 = pred_y_norm_kp1 * Y_std + Y_mean
                actual_vcorr_kp1 = actual_y_norm_kp1 * Y_std + Y_mean if actual_y_norm_kp1 is not None else None
                
                vcorr_error = pred_vcorr_k - actual_vcorr_k
                pred_error = pred_vcorr_kp1 - actual_vcorr_kp1 if actual_vcorr_kp1 is not None else None
                
                if k == 0:
                    print(f"  Step | Pred Y(k) | Actual Y(k) | Pred Vcorr(k) | Actual Vcorr(k) | Error | Pred Vcorr(k+1) | Actual Vcorr(k+1) | Pred Error")
                    print(f"  {'-'*120}")
                
                pred_error_str = f"{pred_error:+.6f}" if pred_error is not None else "N/A"
                actual_vcorr_kp1_str = f"{actual_vcorr_kp1:.6f}" if actual_vcorr_kp1 is not None else "N/A"
                print(f"  {k:4d} | {pred_y_norm_k:8.2f} | {actual_y_norm_k:11.2f} | {pred_vcorr_k:13.6f} | {actual_vcorr_k:14.6f} | {vcorr_error:+6.6f} | "
                      f"{pred_vcorr_kp1:15.6f} | {actual_vcorr_kp1_str:17s} | {pred_error_str}")

    # Denormalize to get actual Vcorr (same as Neural ODE)
    pred_vcorr_denorm = pred_y_norm.detach().cpu().numpy() * Y_std + Y_mean
    return pred_vcorr_denorm


def run_dnn_benchmark_inference(
    dict_list: Sequence[dict],
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Union[str, float, np.ndarray]]], Dict[str, Union[float, Dict]]]:
    """
    Run autoregressive inference for the DNN benchmark over a list of profiles.

    Returns:
        results: list of dicts containing time-series outputs per profile
        metrics: dict with aggregate statistics and model metadata
    """
    device = _resolve_device(device)
    model, metadata = load_dnn_benchmark_model(model_path, device=device)

    if verbose:
        print(f"\n[TEST] Loading checkpoint: {model_path}")
        if metadata:
            print("  --- Checkpoint Metadata ---")
            for key in sorted(metadata.keys()):
                print(f"  {key}: {metadata[key]}")
        else:
            print("  (no metadata found in checkpoint)")
        print("=" * 60)

    total_profiles = len(dict_list)
    batch_size = min(batch_size or total_profiles or 1, total_profiles or 1)

    features_list: List[np.ndarray] = []
    v_spme_list: List[np.ndarray] = []
    v_meas_list: List[np.ndarray] = []
    time_list: List[Optional[np.ndarray]] = []
    soc_list: List[Optional[np.ndarray]] = []
    profile_names: List[str] = []
    y_mean_list: List[float] = []
    y_std_list: List[float] = []
    valid_lengths: List[int] = []

    for idx, profile in enumerate(dict_list):
        features, _, v_spme, v_meas = _extract_features_and_targets(profile)
        if "Y_mean" not in profile or "Y_std" not in profile:
            raise KeyError("Profile must contain 'Y_mean' and 'Y_std' for denormalization.")
        y_mean = float(profile["Y_mean"])
        y_std = float(profile["Y_std"])

        features_list.append(features)
        v_spme_list.append(v_spme)
        v_meas_list.append(v_meas)
        time_list.append(_resolve_array(profile, "time"))
        soc_list.append(_resolve_array(profile, "SOC"))
        profile_names.append(_resolve_profile_name(profile, fallback=f"profile_{idx}"))
        y_mean_list.append(y_mean)
        y_std_list.append(y_std)
        valid_lengths.append(features.shape[0])

    if verbose:
        print(f"[TEST] Total profiles to evaluate: {total_profiles}")
        max_steps = max(valid_lengths) if valid_lengths else 0
        print(f"✓ Batch inputs set: {total_profiles} profiles, {max_steps} time steps")

    progress_step = max(1, math.ceil(total_profiles * 0.1)) if total_profiles else 1
    next_progress = progress_step
    processed_profiles = 0

    results: List[Dict[str, Union[str, float, np.ndarray]]] = []
    total_squared_error = 0.0
    total_points = 0
    rmse_list: List[float] = []
    rmse_vcorr_list: List[float] = []

    for start in range(0, total_profiles, batch_size):
        end = min(start + batch_size, total_profiles)
        batch_indices = list(range(start, end))
        batch_count = len(batch_indices)
        batch_max_len = max(valid_lengths[i] for i in batch_indices)

        features_batch = torch.zeros(batch_count, batch_max_len, 6, dtype=torch.float32, device=device)
        valid_mask = torch.zeros(batch_count, batch_max_len, dtype=torch.bool, device=device)

        for local_idx, global_idx in enumerate(batch_indices):
            feat_np = features_list[global_idx]
            feat_len = feat_np.shape[0]
            features_batch[local_idx, :feat_len] = torch.from_numpy(feat_np).to(device)
            valid_mask[local_idx, :feat_len] = True

        pred_y_norm_batch = _dnn_autoregressive_rollout_inference_batch(
            model, features_batch, valid_mask
        )  # [batch_count, batch_max_len]

        for local_idx, global_idx in enumerate(batch_indices):
            valid_len = valid_lengths[global_idx]
            y_mean = y_mean_list[global_idx]
            y_std = y_std_list[global_idx]
            pred_norm = pred_y_norm_batch[local_idx, :valid_len].cpu().numpy()
            pred_vcorr = pred_norm * y_std + y_mean

            v_spme = v_spme_list[global_idx][:valid_len]
            v_meas = v_meas_list[global_idx][:valid_len]
            v_pred = v_spme + pred_vcorr
            squared_error = float(np.sum((v_pred - v_meas) ** 2))
            rmse = float(np.sqrt(np.mean((v_pred - v_meas) ** 2)))

            actual_vcorr = v_meas - v_spme
            rmse_vcorr = float(np.sqrt(np.mean((pred_vcorr - actual_vcorr) ** 2)))

            total_squared_error += squared_error
            total_points += valid_len
            processed_profiles += 1
            rmse_list.append(rmse)
            rmse_vcorr_list.append(rmse_vcorr)

            if verbose and processed_profiles >= next_progress:
                pct = min(100, int(round(processed_profiles / total_profiles * 100)))
                print(f"[TEST] Progress: {processed_profiles}/{total_profiles} profiles ({pct}%)")
                next_progress += progress_step

            results.append(
                {
                    "profile_index": global_idx,
                    "profile_name": profile_names[global_idx],
                    "time": time_list[global_idx],
                    "V_meas": v_meas,
                    "V_spme": v_spme,
                    "Vcorr_pred": pred_vcorr,
                    "SOC": soc_list[global_idx],
                    "rmse": rmse,
                }
            )

    overall_rmse = (
        float(np.sqrt(total_squared_error / total_points)) if total_points > 0 else float("nan")
    )
    avg_rmse_mV = float(np.mean(rmse_list) * 1e3) if rmse_list else float("nan")
    median_rmse_mV = float(np.median(rmse_list) * 1e3) if rmse_list else float("nan")
    avg_rmse_vcorr_mV = float(np.mean(rmse_vcorr_list) * 1e3) if rmse_vcorr_list else float("nan")
    median_rmse_vcorr_mV = float(np.median(rmse_vcorr_list) * 1e3) if rmse_vcorr_list else float("nan")

    metrics: Dict[str, Union[float, Dict]] = {
        "overall_rmse": overall_rmse,
        "avg_rmse_mV": avg_rmse_mV,
        "median_rmse_mV": median_rmse_mV,
        "avg_rmse_vcorr_mV": avg_rmse_vcorr_mV,
        "median_rmse_vcorr_mV": median_rmse_vcorr_mV,
        "model_metadata": metadata,
    }

    if verbose:
        if results:
            last_rmse = rmse_list[-1] * 1e3
            print(f"[TEST] Profiles 0-{total_profiles-1}: last RMSE = {last_rmse:.2f} mV")
        print(f"[TEST] Completed {total_profiles} profiles (batch_size={batch_size}, solver=autoregressive)")
        print(
            f"[TEST] RMSE Vcorr -> avg: {avg_rmse_vcorr_mV:.2f} mV, "
            f"median: {median_rmse_vcorr_mV:.2f} mV"
        )
        print(
            f"[TEST] RMSE Vtotal -> avg: {avg_rmse_mV:.2f} mV, "
            f"median: {median_rmse_mV:.2f} mV"
        )

    return results, metrics


