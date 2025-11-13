"""
Benchmark models for one-step-ahead Vcorr prediction.

Each model consumes the same feature vector that feeds the Neural ODE
(`V_ref`, `ocv`, `Vcorr`, `SOC`, `I`, `T`) at time step *k* and predicts
`Vcorr(k+1)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEFAULT_INPUT_DIM = 6  # [V_ref, ocv, Vcorr, SOC, I, T]


def _build_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    activation: nn.Module = nn.Tanh(),
) -> nn.Sequential:
    """
    Build a multi-layer perceptron (MLP).
    
    Uses PyTorch's default initialization (Kaiming uniform for Linear layers).
    This matches the previous working setup.
    """
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    net = nn.Sequential(*layers)
    
    # Use PyTorch's default initialization (Kaiming uniform for Linear layers)
    # This matches the previous working setup where no explicit initialization was used
    
    return net


class DNNBenchmark(nn.Module):
    """
    Feed-forward network that predicts Vcorr(k+1) directly in denormalized space (physical units, mV).
    This approach worked well previously despite teacher forcing vs autoregressive mismatch.
    Uses mixed normalization: V_ref, ocv, I, T are normalized, but Vcorr is denormalized (mV).
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM):
        super().__init__()
        self.net = _build_mlp(
            input_dim=input_dim,
            hidden_dims=(32, 32, 32, 16),  # Original architecture (matches saved checkpoint)
            output_dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., input_dim) containing [V_ref, ocv, Vcorr, SOC, I, T]
               where V_ref, ocv, I, T are normalized, but Vcorr is denormalized (mV)
        Returns:
            (..., 1) predicted Vcorr(k+1) in denormalized space (physical units, mV)
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
    
    Features use mixed normalization (previous working approach):
    - V_ref, ocv, I, T: normalized (different scales)
    - Vcorr: denormalized (physical units, mV) - small range, stable
    - SOC: already normalized (0~1)
    
    Returns denormalized Vcorr for training targets and RMSE calculation.
    """
    V_ref = _resolve_array(profile, "V_ref")
    if V_ref is None:
        V_ref = _resolve_array(profile, "V_meas")
    ocv = _resolve_array(profile, "ocv")
    soc = _resolve_array(profile, "SOC")
    current = _resolve_array(profile, "I")
    temperature = _resolve_array(profile, "T")

    v_spme = _resolve_array(profile, "V_spme")
    v_meas = _resolve_array(profile, "V_meas")

    # Get denormalized Vcorr for features (previous working approach)
    vcorr = _resolve_array(profile, "Vcorr")
    if vcorr is None:
        # Try to compute from V_meas/V_spme
        if v_meas is not None and v_spme is not None:
            vcorr = v_meas - v_spme
        else:
            # Try to get from normalized Y and denormalize
            Y_norm = _resolve_array(profile, "Y")
            if Y_norm is not None and "Y_mean" in profile and "Y_std" in profile:
                Y_mean = float(profile["Y_mean"])
                Y_std = float(profile["Y_std"])
                vcorr = Y_norm * Y_std + Y_mean  # Denormalize
            else:
                raise KeyError(
                    "Profile must contain 'Vcorr', 'Y' with 'Y_mean' and 'Y_std', "
                    "or both 'V_meas' and 'V_spme'."
                )

    if any(arr is None for arr in (V_ref, ocv, soc, current, temperature, v_spme, v_meas, vcorr)):
        missing = [
            name
            for name, arr in zip(
                ["V_ref/V_meas", "ocv", "SOC", "I", "T", "V_spme", "V_meas", "Vcorr"],
                (V_ref, ocv, soc, current, temperature, v_spme, v_meas, vcorr),
            )
            if arr is None
        ]
        raise KeyError(f"Missing required arrays: {missing}")

    min_len = min(
        len(V_ref), len(ocv), len(vcorr), len(soc), len(current), len(temperature), len(v_spme), len(v_meas)
    )
    if min_len < 2:
        raise ValueError("Each profile must contain at least two time steps.")

    # Features use denormalized Vcorr (previous working approach)
    # Other features (V_ref, ocv, I, T) are normalized, but Vcorr is denormalized
    features = np.stack(
        [
            V_ref[:min_len],
            ocv[:min_len],
            vcorr[:min_len],  # Denormalized Vcorr (physical units, mV)
            soc[:min_len],
            current[:min_len],
            temperature[:min_len],
        ],
        axis=-1,
    ).astype(np.float32)
    vcorr = vcorr[:min_len].astype(np.float32)
    v_spme = v_spme[:min_len].astype(np.float32)
    v_meas = v_meas[:min_len].astype(np.float32)
    return features, vcorr, v_spme, v_meas


class SingleStepDataset(Dataset):
    """
    Dataset for one-step predictors such as DNNBenchmark.
    
    Uses mixed normalization (previous working approach):
    - Input features: V_ref, ocv, I, T normalized; Vcorr denormalized (physical units)
    - Targets: denormalized Vcorr(k+1) (physical units, mV)
    """

    def __init__(self, profiles: Sequence[dict]):
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        v_spme_next: List[np.ndarray] = []
        v_meas_next: List[np.ndarray] = []
        
        for profile_idx, profile in enumerate(profiles):
            features, vcorr, v_spme, v_meas = _extract_features_and_targets(profile)
            
            # Features contain denormalized Vcorr at position 2
            # Targets should be denormalized Vcorr(k+1)
            # features: [V_ref, ocv, Vcorr, SOC, I, T] where Vcorr is denormalized
            # targets: Vcorr(k+1) denormalized
            
            xs.append(features[:-1])
            ys.append(vcorr[1:, None])  # Denormalized Vcorr(k+1) as target
            v_spme_next.append(v_spme[1:, None])
            v_meas_next.append(v_meas[1:, None])

        if not xs:
            raise ValueError("No samples generated – check input profiles.")

        self.inputs = torch.from_numpy(np.concatenate(xs, axis=0))
        self.targets = torch.from_numpy(np.concatenate(ys, axis=0))
        self.v_spme_next = torch.from_numpy(np.concatenate(v_spme_next, axis=0))
        self.v_meas_next = torch.from_numpy(np.concatenate(v_meas_next, axis=0))

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
    training_batch_size: Optional[int],
    num_workers: int,
    *,
    val_dict_list: Optional[Sequence[dict]] = None,
    pretrained_model_path: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    clip_grad_norm: float = 50.0,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-10,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    verbose: bool = True,
) -> BenchmarkTrainingSummary:
    """
    Train the DNN benchmark model using the provided profile dict list.
    
    Args:
        train_dict_list: List of dicts for training
        val_dict_list: Optional list of dicts for validation. If provided, validation
                      will be performed each epoch and used for model selection and early stopping.
    """
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = _resolve_device(device)

    # Training dataset and loader
    if verbose:
        print(f"[DNN] Building training dataset from {len(train_dict_list)} profiles...")
        sys.stdout.flush()  # Force flush to show progress immediately
    
    train_dataset = build_dnn_dataset(train_dict_list)
    effective_batch_size = training_batch_size if training_batch_size is not None else len(train_dataset)
    drop_last = training_batch_size is not None
    num_batches = len(train_dataset) // effective_batch_size + (1 if len(train_dataset) % effective_batch_size != 0 else 0)
    
    if verbose:
        print(f"[DNN] Training dataset size: {len(train_dataset)} samples")
        print(f"[DNN] Batch size: {effective_batch_size}, Num batches: {num_batches}")
        sys.stdout.flush()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    
    # Note: Using denormalized Vcorr for targets (previous working approach)
    # No need for Y_std_avg since we work in physical units
    
    # Validation dataset and loader (if provided)
    val_loader = None
    if val_dict_list is not None and len(val_dict_list) > 0:
        if verbose:
            print(f"[DNN] Building validation dataset from {len(val_dict_list)} profiles...")
            sys.stdout.flush()
        val_dataset = build_dnn_dataset(val_dict_list)
        if verbose:
            print(f"[DNN] Validation dataset size: {len(val_dataset)} samples")
            sys.stdout.flush()
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
    elif val_dict_list is not None and len(val_dict_list) == 0:
        if verbose:
            print(f"[DNN] Validation dataset is empty (len=0), skipping validation.")
            sys.stdout.flush()

    if verbose:
        print(f"[DNN] Initializing model...")
        sys.stdout.flush()
    model = DNNBenchmark().to(device)
    _load_pretrained_state(model, pretrained_model_path)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        threshold=1e-3,
        threshold_mode="rel",
    )
    criterion = torch.nn.MSELoss()
    
    if verbose:
        print(f"[DNN] Starting training for {num_epochs} epochs...")
        print(f"[DNN] Device: {device}, Learning rate: {lr:.6f}")
        print("=" * 80)
        sys.stdout.flush()

    best_rmse = float("inf")  # RMSE in mV (denormalized space)
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
    use_validation = val_loader is not None

    for epoch in range(1, num_epochs + 1):
        # ===== Training Phase =====
        model.train()
        train_running_loss = 0.0
        train_sample_count = 0
        grad_norm_before = 0.0
        grad_norm_after = 0.0
        
        if verbose and epoch == 1:
            print(f"[DNN] Starting epoch {epoch}/{num_epochs}...")
            sys.stdout.flush()

        for batch_idx, (xb, yb, v_spme_next, v_meas_next) in enumerate(train_loader):
            # Show progress for first epoch (first few batches)
            if verbose and epoch == 1 and batch_idx < 3:
                total_batches = len(train_loader)
                print(f"[DNN] Processing batch {batch_idx + 1}/{total_batches}...")
                sys.stdout.flush()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # Direct prediction of Vcorr(k+1) in denormalized space (previous working approach)
            preds = model(xb)  # Vcorr(k+1) prediction in denormalized space (mV)
            
            # Debug: check Y (normalized Vcorr) range in flat OCV region (first batch of first epoch only)
            if epoch == 1 and batch_idx == 0:
                # Convert to Y (normalized Vcorr) for analysis
                # Y = (Vcorr - Y_mean) / Y_std
                # Get Y_mean and Y_std from profile (assume same for all in batch, use first profile's values)
                # Note: xb[:, 2] is Vcorr (denormalized, mV), yb is Vcorr(k+1) (denormalized, mV)
                vcorr_input = xb[:, 2].cpu().numpy()  # Vcorr(k) denormalized
                vcorr_target = yb.cpu().numpy().flatten()  # Vcorr(k+1) denormalized
                soc_values = xb[:, 3].cpu().numpy()  # SOC
                
                # Assume Y_mean and Y_std (from df2dict: Y_mean=-0.015, Y_std=0.004)
                Y_mean = -0.015
                Y_std = 0.004
                
                # Convert to Y (normalized)
                Y_input = (vcorr_input - Y_mean) / Y_std
                Y_target = (vcorr_target - Y_mean) / Y_std
                
                # Flat OCV region: SOC = [0.11, 0.96]
                flat_mask = (soc_values >= 0.11) & (soc_values <= 0.96)
                Y_flat_input = Y_input[flat_mask]
                Y_flat_target = Y_target[flat_mask]
                
                print(f"[DNN] First batch:")
                print(f"  Input Vcorr(k): [{vcorr_input.min():.3f}, {vcorr_input.max():.3f}] mV")
                print(f"  Target Vcorr(k+1): [{vcorr_target.min():.3f}, {vcorr_target.max():.3f}] mV")
                if len(Y_flat_input) > 0:
                    print(f"  Y (norm) in flat region [11-96% SOC]:")
                    print(f"    Input Y(k): [{Y_flat_input.min():.2f}, {Y_flat_input.max():.2f}], std: {Y_flat_input.std():.2f}")
                    print(f"    Target Y(k+1): [{Y_flat_target.min():.2f}, {Y_flat_target.max():.2f}], std: {Y_flat_target.std():.2f}")
                print(f"  Predicted Vcorr(k+1): [{preds.min():.3f}, {preds.max():.3f}] mV, Loss: {criterion(preds, yb).item():.6f}")
            
            loss = criterion(preds, yb)  # Loss in denormalized space (physical units, mV)
            optimizer.zero_grad()
            loss.backward()

            grad_before = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_before += param.grad.data.norm(2).item() ** 2
            grad_norm_before += grad_before ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            grad_after = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_after += param.grad.data.norm(2).item() ** 2
            grad_norm_after += grad_after ** 0.5

            optimizer.step()

            train_running_loss += loss.item() * xb.size(0)
            train_sample_count += xb.size(0)

        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        
        # Calculate RMSE in denormalized space (physical units, mV)
        train_rmse_mV = (train_running_loss / len(train_loader.dataset)) ** 0.5 * 1000  # Convert to mV

        avg_grad_before = grad_norm_before / len(train_loader)
        avg_grad_after = grad_norm_after / len(train_loader)

        # ===== Validation Phase =====
        val_epoch_loss = None
        val_epoch_rmse_mV = None
        if use_validation:
            model.eval()
            val_running_loss = 0.0
            
            with torch.no_grad():
                # Use batch processing with teacher forcing (fast, same as training)
                # This is faster than autoregressive rollout and gives consistent validation metrics
                for xb, yb, v_spme_next, v_meas_next in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    
                    # Direct prediction of Vcorr(k+1) in denormalized space (same as training)
                    preds = model(xb)  # Vcorr(k+1) prediction in denormalized space (mV)
                    loss = criterion(preds, yb)  # Loss in denormalized space (physical units, mV)
                    
                    val_running_loss += loss.item() * xb.size(0)
            
            # Calculate RMSE in denormalized space (physical units, mV)
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_epoch_rmse_mV = (val_epoch_loss) ** 0.5 * 1000  # Convert to mV
        
        # Use validation metrics for scheduler and model selection if available, otherwise use training metrics
        monitor_loss = val_epoch_loss if use_validation else train_epoch_loss
        scheduler.step(monitor_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_epoch_loss)
        history["train_rmse_mV"].append(train_rmse_mV)  # RMSE in mV (denormalized space)
        history["val_loss"].append(val_epoch_loss if val_epoch_loss is not None else float("nan"))
        history["val_rmse_mV"].append(val_epoch_rmse_mV if val_epoch_rmse_mV is not None else float("nan"))  # RMSE in mV (denormalized space)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grad_before"].append(avg_grad_before)
        history["grad_after"].append(avg_grad_after)

        # Print every epoch if verbose
        if verbose:
            if use_validation:
                print(
                    f"[DNN] epoch {epoch}/{num_epochs}, "
                    f"train_loss={train_epoch_loss:.3e}, train_RMSE={train_rmse_mV:.2f} mV, "
                    f"val_loss={val_epoch_loss:.3e}, val_RMSE={val_epoch_rmse_mV:.2f} mV, "
                    f"LR={optimizer.param_groups[0]['lr']:.3e}, "
                    f"Grad={avg_grad_before:.3e}->{avg_grad_after:.3e}"
                )
            else:
                print(
                    f"[DNN] epoch {epoch}/{num_epochs}, "
                    f"loss={train_epoch_loss:.3e}, "
                    f"RMSE={train_rmse_mV:.2f} mV, "
                    f"LR={optimizer.param_groups[0]['lr']:.3e}, "
                    f"Grad={avg_grad_before:.3e}->{avg_grad_after:.3e}"
                )

        # Use RMSE in denormalized space (physical units, mV)
        # All calculations in denormalized space
        
        # Best model selection: use validation if available (to prevent overfitting)
        if use_validation and val_epoch_rmse_mV is not None:
            best_monitor_rmse_mV = val_epoch_rmse_mV  # Already in mV
            best_metric_name = "val_RMSE"
        else:
            best_monitor_rmse_mV = train_rmse_mV  # Already in mV
            best_metric_name = "RMSE"
        
        # Early stopping: use training RMSE (more stable, reflects actual learning progress)
        early_stop_monitor_rmse_mV = train_rmse_mV  # Always use training RMSE for early stopping
        
        # Update best model based on validation (if available) or training
        rmse_window.append(early_stop_monitor_rmse_mV)  # Early stopping uses training RMSE
        if best_monitor_rmse_mV < best_rmse:
            best_rmse = best_monitor_rmse_mV
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

    save_path = Path(save_path) if save_path is not None else _default_artifact_path("dnn")
    payload = {
        "model_kind": "dnn",
        "state_dict": best_state_dict,
        "model_kwargs": {"input_dim": DEFAULT_INPUT_DIM},
        "architecture": repr(model),
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": training_batch_size or len(train_loader.dataset),
        "num_workers": num_workers,
        "best_epoch": best_epoch,
        "best_rmse_mV": best_rmse,  # Already in mV (denormalized space)
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
        batch_size=training_batch_size or len(train_loader.dataset),
        num_workers=num_workers,
        best_epoch=best_epoch,
        best_rmse_mV=best_rmse,  # Already in mV (denormalized space)
        history=history,
    )

    if verbose:
        print(f"[DNN] Best RMSE = {summary.best_rmse_mV:.2f} mV at epoch {summary.best_epoch}")
        print(f"[DNN] Architecture:")
        print(summary.architecture)
        print(f"[DNN] Saved best model to: {summary.state_dict_path}")

    return summary


def load_dnn_benchmark_model(
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[DNNBenchmark, Dict[str, Union[int, float, str, Dict[str, List[float]]]]]:
    """
    Load a serialized DNN benchmark checkpoint and return the model plus metadata.
    """
    device = _resolve_device(device)
    model = DNNBenchmark().to(device)
    payload = torch.load(model_path, map_location=device)
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


def _dnn_autoregressive_rollout(
    model: DNNBenchmark,
    features: np.ndarray,
    device: torch.device,
    debug: bool = False,
) -> np.ndarray:
    """
    Perform an autoregressive rollout over a single profile.

    Args:
        model: trained DNNBenchmark
        features: np.ndarray shape [T, 6] as returned by _extract_features_and_targets
                 Contains mixed normalization: [V_ref, ocv, Vcorr, SOC, I, T]
                 where V_ref, ocv, I, T are normalized, but Vcorr is denormalized (mV)
        device: torch device
        debug: if True, print debug info for first 3 steps
    Returns:
        np.ndarray shape [T] with predicted Vcorr in physical units (denormalized, mV)
    """
    seq_len = features.shape[0]
    static_features = torch.from_numpy(features).to(device, dtype=torch.float32)
    pred_vcorr = torch.zeros(seq_len, dtype=torch.float32, device=device)
    pred_vcorr[0] = static_features[0, 2]  # Initial condition: denormalized Vcorr (mV)

    with torch.no_grad():
        for k in range(seq_len - 1):
            input_vec = torch.stack(
                [
                    static_features[k, 0],  # V_ref (normalized)
                    static_features[k, 1],  # ocv (normalized)
                    pred_vcorr[k],           # autoregressive Vcorr(k) (denormalized, mV)
                    static_features[k, 3],  # SOC (normalized)
                    static_features[k, 4],  # current (normalized)
                    static_features[k, 5],  # temperature (normalized)
                ]
            ).unsqueeze(0)
            # Direct prediction of Vcorr(k+1) in denormalized space (previous working approach)
            pred_vcorr_kp1 = model(input_vec).squeeze(0).squeeze(0)
            # Debug: check for NaN or Inf
            if torch.isnan(pred_vcorr_kp1) or torch.isinf(pred_vcorr_kp1):
                raise ValueError(f"NaN or Inf detected in pred_vcorr at k={k}")
            
            # Direct prediction: Vcorr(k+1) = predicted directly (denormalized, mV)
            pred_vcorr[k + 1] = pred_vcorr_kp1
            
            # Debug: print first few steps if requested (only for first profile)
            if debug and k < 3:
                actual_vcorr_k = static_features[k, 2].item()
                actual_vcorr_kp1 = static_features[k+1, 2].item() if k+1 < seq_len else None
                pred_vcorr_k = pred_vcorr[k].item()
                pred_vcorr_kp1 = pred_vcorr_kp1.item()
                vcorr_error = pred_vcorr_k - actual_vcorr_k
                pred_error = pred_vcorr_kp1 - actual_vcorr_kp1 if actual_vcorr_kp1 is not None else None
                
                if k == 0:
                    print(f"  Step | Pred Vcorr(k) | Actual Vcorr(k) | Error | Pred Vcorr(k+1) | Actual Vcorr(k+1) | Pred Error")
                    print(f"  {'-'*95}")
                
                pred_error_str = f"{pred_error:+.6f}" if pred_error is not None else "N/A"
                actual_vcorr_kp1_str = f"{actual_vcorr_kp1:.6f}" if actual_vcorr_kp1 is not None else "N/A"
                print(f"  {k:4d} | {pred_vcorr_k:13.6f} | {actual_vcorr_k:14.6f} | {vcorr_error:+6.6f} | "
                      f"{pred_vcorr_kp1:15.6f} | {actual_vcorr_kp1_str:17s} | {pred_error_str}")

    # Return denormalized Vcorr (already in physical units, mV)
    pred_vcorr_np = pred_vcorr.detach().cpu().numpy()
    return pred_vcorr_np


def run_dnn_benchmark_inference(
    dict_list: Sequence[dict],
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[List[Dict[str, Union[str, float, np.ndarray]]], Dict[str, Union[float, Dict]]]:
    """
    Run autoregressive inference for the DNN benchmark over a list of profiles.

    Returns:
        results: list of dicts containing time-series outputs per profile
        metrics: dict with aggregate statistics and model metadata
    """
    device = _resolve_device(device)
    model, metadata = load_dnn_benchmark_model(model_path, device=device)

    results: List[Dict[str, Union[str, float, np.ndarray]]] = []
    total_squared_error = 0.0
    total_points = 0

    for idx, profile in enumerate(dict_list):
        features, _, v_spme, v_meas = _extract_features_and_targets(profile)
        
        # Debug only for first profile
        debug = (idx == 0)
        if debug:
            print(f"[DEBUG] Profile #{idx} (first 3 steps):")
        # Note: features already contains denormalized Vcorr at position 2
        pred_vcorr = _dnn_autoregressive_rollout(model, features, device, debug=debug)
        if debug:
            print()  # Empty line after debug output

        time_values = _resolve_array(profile, "time")
        soc_values = _resolve_array(profile, "SOC")
        profile_name = _resolve_profile_name(profile, fallback=f"profile_{idx}")

        # Calculate V_pred = V_spme + Vcorr_pred (denormalized)
        # pred_vcorr is already denormalized from _dnn_autoregressive_rollout
        v_pred = v_spme + pred_vcorr  # V_pred = V_spme + Vcorr_pred (denormalized)
        squared_error = np.sum((v_pred - v_meas) ** 2)
        rmse = float(np.sqrt(np.mean((v_pred - v_meas) ** 2)))

        total_squared_error += squared_error
        total_points += len(v_meas)
        
        # Print RMSE for first few profiles for debugging
        if idx < 3:
            print(f"  Profile #{idx} RMSE: {rmse*1e3:.2f} mV")

        results.append(
            {
                "profile_index": idx,
                "profile_name": profile_name,
                "time": time_values,
                "V_meas": v_meas,  # Already denormalized (original)
                "V_spme": v_spme,  # Already denormalized
                "Vcorr_pred": pred_vcorr,  # Only Vcorr prediction (denormalized), not including v_spme
                "SOC": soc_values,
                "rmse": rmse,
            }
        )

    overall_rmse = (
        float(np.sqrt(total_squared_error / total_points)) if total_points > 0 else float("nan")
    )
    metrics: Dict[str, Union[float, Dict]] = {
        "overall_rmse": overall_rmse,
        "model_metadata": metadata,
    }
    return results, metrics


