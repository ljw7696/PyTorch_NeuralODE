import copy
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BatteryGRUWrapper(nn.Module):
    """
    Sequence model wrapper for battery voltage correction.

    This mirrors the API style of `BatteryODEWrapper` but replaces the ODE block
    with a GRU that directly predicts Vcorr trajectories from the same inputs.
    
    Architecture: Bidirectional GRU(32) → Unidirectional GRU(24) → Dense(16) → Dense(8) → Linear(1)
    
    Input: [V_spme_norm, ocv, Vcorr, SOC, I, T] (6 features)
    Output: Vcorr prediction (normalized)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 12,
        num_layers: int = 1,
        dropout: float = 0.1,
        device: Union[str, torch.device] = "cpu",
        # New architecture parameters
        gru1_hidden: int = 32,
        gru2_hidden: int = 24,
        dense1_hidden: int = 16,
        dense2_hidden: int = 8,
    ):
        super().__init__()
        self.device = torch.device(device)

        # Store architecture info for checkpoint
        self.architecture_config = {
            "input_size": input_size,
            "gru1_hidden": gru1_hidden,
            "gru2_hidden": gru2_hidden,
            "dense1_hidden": dense1_hidden,
            "dense2_hidden": dense2_hidden,
        }

        # Bidirectional GRU layer (skip if gru1_hidden == 0)
        self.use_gru1 = gru1_hidden > 0
        if self.use_gru1:
            self.gru1 = nn.GRU(
            input_size=input_size,
                hidden_size=gru1_hidden,
                num_layers=1,
                bidirectional=True,
            batch_first=True,
        )
        else:
            self.gru1 = None
        
        # Unidirectional GRU layer (skip if gru2_hidden == 0)
        self.use_gru2 = gru2_hidden > 0
        if self.use_gru2:
            # gru2_input_size: If None, use gru1 output (gru1_hidden * 2), otherwise use specified value
            # This allows backward compatibility with old single-GRU checkpoints
            gru2_input_size = getattr(self, '_gru2_input_size_override', None)
            if gru2_input_size is None:
                if self.use_gru1:
                    gru2_input_size = gru1_hidden * 2  # Bidirectional output
                else:
                    gru2_input_size = input_size  # Skip gru1, use input directly
            self.gru2 = nn.GRU(
                input_size=gru2_input_size,
                hidden_size=gru2_hidden,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )
        else:
            self.gru2 = None
        
        # Determine input size for dense layers
        if self.use_gru2:
            dense_input_size = gru2_hidden
        elif self.use_gru1:
            dense_input_size = gru1_hidden * 2  # Bidirectional output
        else:
            dense_input_size = input_size
        
        # Dense layers (skip if size == 0)
        self.use_dense1 = dense1_hidden > 0
        if self.use_dense1:
            self.dense1 = nn.Linear(dense_input_size, dense1_hidden)
        else:
            self.dense1 = None
        
        self.use_dense2 = dense2_hidden > 0
        if self.use_dense2:
            dense2_input_size = dense1_hidden if self.use_dense1 else dense_input_size
            self.dense2 = nn.Linear(dense2_input_size, dense2_hidden)
        else:
            self.dense2 = None
        
        # Final head layer
        head_input_size = dense2_hidden if self.use_dense2 else (dense1_hidden if self.use_dense1 else dense_input_size)
        self.head = nn.Linear(head_input_size, 1)
        
        # Activation (no dropout)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)
        self.hidden_state: Optional[torch.Tensor] = None
        
        # Print architecture
        print(f"[BatteryGRUWrapper] Architecture initialized:")
        print(f"  Input size: {input_size}")
        if self.use_gru1:
            print(f"  Bidirectional GRU1: {input_size} → {gru1_hidden} (output: {gru1_hidden * 2})")
        else:
            print(f"  Bidirectional GRU1: SKIPPED")
        if self.use_gru2:
            gru2_in = gru1_hidden * 2 if self.use_gru1 else input_size
            print(f"  Unidirectional GRU2: {gru2_in} → {gru2_hidden}")
        else:
            print(f"  Unidirectional GRU2: SKIPPED")
        if self.use_dense1:
            dense1_in = gru2_hidden if self.use_gru2 else (gru1_hidden * 2 if self.use_gru1 else input_size)
            print(f"  Dense1: {dense1_in} → {dense1_hidden}")
        else:
            print(f"  Dense1: SKIPPED")
        if self.use_dense2:
            dense2_in = dense1_hidden if self.use_dense1 else (gru2_hidden if self.use_gru2 else (gru1_hidden * 2 if self.use_gru1 else input_size))
            print(f"  Dense2: {dense2_in} → {dense2_hidden}")
        else:
            print(f"  Dense2: SKIPPED")
        head_in = dense2_hidden if self.use_dense2 else (dense1_hidden if self.use_dense1 else (gru2_hidden if self.use_gru2 else (gru1_hidden * 2 if self.use_gru1 else input_size)))
        print(f"  Linear: {head_in} → 1")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.7)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0.0)

    def reset_hidden_state(self, batch_size: int = 1) -> None:
        """Reset hidden states for both GRU layers"""
        if self.use_gru1:
            num_layers1 = self.gru1.num_layers
            hidden_size1 = self.gru1.hidden_size
            # Bidirectional: 2 directions
            self.hidden_state_gru1 = torch.zeros(
                num_layers1 * 2, batch_size, hidden_size1, device=self.device
            )
        else:
            # Dummy hidden state if gru1 is not used
            self.hidden_state_gru1 = torch.zeros(2, batch_size, 1, device=self.device)
        
        if self.use_gru2:
            num_layers2 = self.gru2.num_layers
            hidden_size2 = self.gru2.hidden_size
            self.hidden_state_gru2 = torch.zeros(
                num_layers2, batch_size, hidden_size2, device=self.device
            )
        else:
            # Dummy hidden state if gru2 is not used
            self.hidden_state_gru2 = torch.zeros(1, batch_size, 1, device=self.device)
        
        # For backward compatibility
        self.hidden_state = self.hidden_state_gru1

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor [batch, seq_len, input_size]
            hidden: Optional tuple of (gru1_hidden, gru2_hidden) for state continuation
        Returns:
            preds: Tensor [batch, seq_len, 1]
            hidden: Tuple of (gru1_hidden, gru2_hidden) final hidden states
        """
        batch_size, seq_len, _ = inputs.shape
        x = inputs
        
        # Process through GRU layers
        gru1_hidden = None
        gru2_hidden = None
        
        if self.use_gru1:
            # GRU1: Bidirectional
            if hidden is not None:
                gru1_hidden_prev, _ = hidden
                x, gru1_hidden = self.gru1(x, gru1_hidden_prev)
            else:
                x, gru1_hidden = self.gru1(x)
            # x: [batch, seq_len, gru1_hidden * 2]
        
        if self.use_gru2:
            # GRU2: Unidirectional
            if hidden is not None:
                _, gru2_hidden_prev = hidden
                x, gru2_hidden = self.gru2(x, gru2_hidden_prev)
            else:
                x, gru2_hidden = self.gru2(x)
            # x: [batch, seq_len, gru2_hidden]
        
        # If no GRU layers, create dummy hidden states
        if gru1_hidden is None:
            if self.use_gru1:
                # Should not happen, but create dummy if needed
                num_layers1 = self.gru1.num_layers
                hidden_size1 = self.gru1.hidden_size
                gru1_hidden = torch.zeros(
                    num_layers1 * 2, batch_size, hidden_size1,
                    device=x.device, dtype=x.dtype
                )
            else:
                # No gru1, create dummy
                gru1_hidden = torch.zeros(2, batch_size, 1, device=x.device, dtype=x.dtype)
        
        if gru2_hidden is None:
            if self.use_gru2:
                # Should not happen, but create dummy if needed
                num_layers2 = self.gru2.num_layers
                hidden_size2 = self.gru2.hidden_size
                gru2_hidden = torch.zeros(
                    num_layers2, batch_size, hidden_size2,
                    device=x.device, dtype=x.dtype
                )
            else:
                # No gru2, create dummy
                gru2_hidden = torch.zeros(1, batch_size, 1, device=x.device, dtype=x.dtype)
        
        # Apply dense layers to each timestep
        # Reshape to [batch * seq_len, feature_dim] for batch processing
        feature_dim = x.shape[-1]
        x_flat = x.reshape(-1, feature_dim)
        
        # Dense layers with ReLU (no dropout)
        if self.use_dense1:
            x_flat = self.relu(self.dense1(x_flat))
        
        if self.use_dense2:
            x_flat = self.relu(self.dense2(x_flat))
        
        # Final prediction
        preds_flat = self.head(x_flat)  # [batch * seq_len, 1]
        
        # Reshape back to [batch, seq_len, 1]
        preds = preds_flat.reshape(batch_size, seq_len, 1)
        
        return preds, (gru1_hidden, gru2_hidden)


def _build_feature_tensor(data_dict: Dict[str, Sequence[float]], device: torch.device):
    required_keys = ["V_spme_norm", "ocv", "SOC", "I", "T", "V_spme"]
    for key in required_keys:
        if key not in data_dict:
            raise KeyError(f"Missing '{key}' in data_dict")

    features = np.stack([np.array(data_dict[k], dtype=np.float32) for k in required_keys], axis=-1)
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device).contiguous()
    targets_tensor = (
        torch.tensor(data_dict["Y"], dtype=torch.float32, device=device).unsqueeze(-1).contiguous()
    )
    return features_tensor.unsqueeze(0), targets_tensor.unsqueeze(0)


def _build_sliding_window_dataset(
    data_dict: Dict[str, Sequence[float]],
    window_len: int,
    predict_delta: bool = False,
) -> TensorDataset:
    if window_len < 1:
        raise ValueError("window_len must be >= 1")

    required_keys = ["V_spme_norm", "ocv", "SOC", "I", "T", "V_spme", "Y"]
    for key in required_keys:
        if key not in data_dict:
            raise KeyError(f"Missing '{key}' in data_dict")

    feature_array = np.stack(
        [
            np.asarray(data_dict["V_spme_norm"], dtype=np.float32),
            np.asarray(data_dict["ocv"], dtype=np.float32),
            np.asarray(data_dict["Y"], dtype=np.float32),
            np.asarray(data_dict["SOC"], dtype=np.float32),
            np.asarray(data_dict["I"], dtype=np.float32),
            np.asarray(data_dict["T"], dtype=np.float32),
        ],
        axis=-1,
    )

    total_steps = feature_array.shape[0]
    if total_steps <= window_len:
        raise ValueError(
            f"Sequence length ({total_steps}) must be greater than window_len ({window_len})."
        )

    sequences: List[np.ndarray] = []
    targets = []
    Y_array = np.asarray(data_dict["Y"], dtype=np.float32)
    for idx in range(window_len, total_steps):
        window = feature_array[idx - window_len : idx]
        target_val = Y_array[idx]
        if predict_delta:
            target_val = target_val - Y_array[idx - 1]
        sequences.append(window.astype(np.float32))
        targets.append(np.array([target_val], dtype=np.float32))

    if not sequences:
        raise ValueError("No windows generated. Check window_len vs sequence length.")

    windows_tensor = torch.from_numpy(np.stack(sequences)).contiguous()
    targets_tensor = torch.from_numpy(np.stack(targets)).contiguous()
    return TensorDataset(windows_tensor, targets_tensor)


def _resolve_array(profile: Dict[str, Any], aliases: Sequence[str]) -> np.ndarray:
    for alias in aliases:
        if alias in profile:
            value = profile[alias]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif hasattr(value, "to_numpy"):
                value = value.to_numpy()
            return np.asarray(value, dtype=np.float32)
    raise KeyError(f"Missing keys {aliases} in profile")


def prepare_profile_for_gru(profile: Dict[str, Any]) -> Dict[str, np.ndarray]:
    prepared = {
        "time": _resolve_array(profile, ["time"]),
        "V_spme_norm": _resolve_array(profile, ["V_spme_norm"]),
        "ocv": _resolve_array(profile, ["ocv", "ocp"]),
        "SOC": _resolve_array(profile, ["SOC", "soc_n", "soc"]),
        "I": _resolve_array(profile, ["I", "current"]),
        "T": _resolve_array(profile, ["T", "temperature"]),
        "V_spme": _resolve_array(profile, ["V_spme", "Vspme"]),
        "Y": _resolve_array(profile, ["Y", "Vcorr_norm"]),
        "V_meas": _resolve_array(profile, ["V_meas", "V_ref", "Vref"]),
    }

    prepared["Y_mean"] = float(np.asarray(profile.get("Y_mean", 0.0)))
    prepared["Y_std"] = float(np.asarray(profile.get("Y_std", 1.0)))
    return prepared


def stitch_profiles_for_gru(profiles: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    concat_keys = ["time", "V_spme_norm", "ocv", "SOC", "I", "T", "V_spme", "Y", "V_meas"]
    stitched = {k: [] for k in concat_keys}
    y_means: List[float] = []
    y_stds: List[float] = []
    time_offset = 0.0

    for profile in profiles:
        prepared = prepare_profile_for_gru(profile)
        local_time = prepared["time"].copy()
        if len(local_time) < 2:
            continue
        local_time = local_time - local_time[0] + time_offset
        stitched["time"].append(local_time)
        for key in concat_keys[1:]:
            stitched[key].append(prepared[key])

        dt = float(np.median(np.diff(local_time))) if len(local_time) > 1 else 1.0
        time_offset = local_time[-1] + dt
        y_means.append(prepared["Y_mean"])
        y_stds.append(prepared["Y_std"])

    merged = {k: np.concatenate(v, axis=0) for k, v in stitched.items() if v}
    merged["Y_mean"] = float(np.mean(y_means)) if y_means else 0.0
    merged["Y_std"] = float(np.mean(y_stds)) if y_stds else 1.0
    return merged


def train_battery_gru(
    train_dict_list: Sequence[Dict[str, Any]],
    num_epochs: int = 200,
    lr: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
    pretrained_model_path: Optional[Union[str, Path]] = None,
    predict_delta: bool = True,
    early_stop_patience: int = 50,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    val_dict_list: Optional[Sequence[Dict[str, Any]]] = None,
    shuffle_profiles: bool = True,
    hidden_size: int = 12,
    num_layers: int = 1,
    batch_size: Optional[int] = None,
    gru1_hidden: int = 32,
    gru2_hidden: int = 24,
    dense1_hidden: int = 16,
    dense2_hidden: int = 8,
    loss_scale: float = 1.0,
    tbptt_length: Optional[int] = 300,
) -> Tuple[BatteryGRUWrapper, Dict[str, Dict[str, Optional[float]]]]:
    """
    Profile-wise autoregressive GRU training.
    
    Args:
        batch_size: Number of profiles to process simultaneously on GPU.
                   If None, processes all profiles in one batch.
                   If specified, profiles are split into sub-batches.
    """
    import sys
    sys.stdout.flush()
    print(f"[train_battery_gru] Starting training...", flush=True)
    print(f"[train_battery_gru] train_dict_list length: {len(train_dict_list)}", flush=True)
    device = torch.device(device)
    print(f"[train_battery_gru] Device: {device}", flush=True)
    # Use new architecture: Bidirectional GRU(gru1_hidden) → Unidirectional GRU(gru2_hidden) → Dense(dense1_hidden→dense2_hidden→1)
    model = BatteryGRUWrapper(
        input_size=6,
        hidden_size=hidden_size,  # Kept for backward compatibility but not used
        num_layers=num_layers,  # Kept for backward compatibility but not used
        device=device,
        gru1_hidden=gru1_hidden,
        gru2_hidden=gru2_hidden,
        dense1_hidden=dense1_hidden,
        dense2_hidden=dense2_hidden,
    ).to(device)
    if pretrained_model_path is not None:
        try:
            checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state)
            if verbose:
                print(f"Loaded pretrained GRU weights from {pretrained_model_path}")
        except Exception as exc:
            print(f"⚠️  Could not load pretrained model: {exc}")

    if verbose:
        import sys
        sys.stdout.flush()
        print("\n" + "=" * 70, flush=True)
        print("GRU Training Configuration", flush=True)
        print("=" * 70, flush=True)
        # Print architecture dynamically based on what's actually used
        arch_parts = []
        if model.use_gru1 and model.gru1 is not None:
            arch_parts.append(f"Bidirectional GRU1({model.gru1.hidden_size})")
        if model.use_gru2 and model.gru2 is not None:
            arch_parts.append(f"Unidirectional GRU2({model.gru2.hidden_size})")
        if model.use_dense1 and model.dense1 is not None:
            arch_parts.append(f"Dense1({model.dense1.out_features})")
        if model.use_dense2 and model.dense2 is not None:
            arch_parts.append(f"Dense2({model.dense2.out_features})")
        arch_parts.append("Linear(1)")
        arch_str = " → ".join(arch_parts)
        print(f"Architecture : {arch_str}", flush=True)
        if model.use_gru1 and model.gru1 is not None:
            print(f"  GRU1 (Bidirectional): {model.gru1.input_size} → {model.gru1.hidden_size} (output: {model.gru1.hidden_size * 2})", flush=True)
        else:
            print(f"  GRU1: SKIPPED", flush=True)
        if model.use_gru2 and model.gru2 is not None:
            print(f"  GRU2 (Unidirectional): {model.gru2.input_size} → {model.gru2.hidden_size}", flush=True)
        else:
            print(f"  GRU2: SKIPPED", flush=True)
        if model.use_dense1 and model.dense1 is not None:
            print(f"  Dense1: {model.dense1.in_features} → {model.dense1.out_features}", flush=True)
        else:
            print(f"  Dense1: SKIPPED", flush=True)
        if model.use_dense2 and model.dense2 is not None:
            print(f"  Dense2: {model.dense2.in_features} → {model.dense2.out_features}", flush=True)
        else:
            print(f"  Dense2: SKIPPED", flush=True)
        print(f"  Linear: {model.head.in_features} → {model.head.out_features}", flush=True)
        print(f"Device       : {device}", flush=True)
        print(f"Epochs       : {num_epochs}", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        # Window len no longer used in pure RNN approach
        print(f"Predict delta: {predict_delta}", flush=True)
        print(f"Batch size   : {batch_size if batch_size else 'all profiles'}", flush=True)
        print(f"Train profiles: {len(train_dict_list)}", flush=True)
        print(f"Val profiles  : {len(val_dict_list) if val_dict_list else 0}", flush=True)
        print(f"Early-stop    : patience={early_stop_patience}, window={early_stop_window}, Δ={early_stop_delta_mV} mV", flush=True)
        print("=" * 70 + "\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-3, threshold_mode="rel", min_lr=1e-7
    )

    history: Dict[str, Dict[str, Optional[float]]] = {"best": {}}
    best_rmse = float("inf")
    best_rmse_mV = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    rmse_window = deque(maxlen=early_stop_window)
    use_validation = val_dict_list is not None and len(val_dict_list) > 0
    previous_best_model_path = None  # Track previous checkpoint to delete it

    model.train()
    import time
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        if shuffle_profiles:
            random.shuffle(train_dict_list)  # type: ignore[arg-type]

        epoch_loss = 0.0
        epoch_steps = 0
        train_rmse_mV_accum: List[float] = []
        grad_norms_before: List[float] = []
        grad_norms_after: List[float] = []

        # Process profiles in batches for GPU efficiency
        if batch_size is None:
            # Process all profiles in one batch
            batch_list = [train_dict_list]
            if verbose and epoch == 0:
                print(f"[Epoch {epoch + 1}] Processing {len(train_dict_list)} profiles in 1 batch...")
        else:
            # Split into sub-batches
            batch_list = [
                train_dict_list[i : i + batch_size]
                for i in range(0, len(train_dict_list), batch_size)
            ]
            if verbose and epoch == 0:
                print(f"[Epoch {epoch + 1}] Processing {len(train_dict_list)} profiles in {len(batch_list)} batches...")
        
        # Track statistics across all batches
        total_valid_profiles_epoch = 0
        total_skipped_profiles_epoch = 0

        for batch_idx, profile_batch in enumerate(batch_list):
            if verbose and epoch == 0 and batch_idx == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(batch_list)} ({len(profile_batch)} profiles)...")
            # Track valid/skipped profiles
            stats = _autoregressive_profile_train_pass_batch(
                model=model,
                profile_list=profile_batch,
                predict_delta=predict_delta,
                device=device,
                batch_size=batch_size,
                verbose=verbose and epoch == 0 and batch_idx == 0,
                loss_scale=loss_scale,
                tbptt_length=tbptt_length,
                optimizer=optimizer,
                grad_norms_before=grad_norms_before,
                grad_norms_after=grad_norms_after,
            )
            
            if stats["num_steps"] == 0:
                continue

            total_valid_profiles_epoch += stats.get("valid_profiles", 0)
            total_skipped_profiles_epoch += stats.get("skipped_profiles", 0)
            epoch_loss += stats.get("total_loss", 0.0)
            epoch_steps += stats["num_steps"]
            train_rmse_mV_accum.append(stats["rmse_mV"])
        
        if verbose and epoch == 0:
            print(f"  [Epoch {epoch + 1}] Valid profiles: {total_valid_profiles_epoch}, Skipped: {total_skipped_profiles_epoch} (length <= 1)")

        if epoch_steps == 0:
            if verbose:
                print("No valid training profiles for this epoch.")
            break

        epoch_loss = epoch_loss / epoch_steps
        train_rmse_mV = float(np.nanmean(train_rmse_mV_accum)) if train_rmse_mV_accum else float("nan")

        val_loss = None
        val_rmse_mV = None
        val_rmse_norm = None

        if use_validation:
            model.eval()  # Set to eval mode for validation
            val_metrics = _evaluate_gru_profiles(
                model=model,
                dict_list=val_dict_list or [],
                predict_delta=predict_delta,
                device=device,
            )
            val_rmse_mV = val_metrics["avg_rmse_vcorr_mV"]
            val_rmse_norm = val_metrics["avg_rmse_norm"]
            val_loss = val_rmse_norm**2 if val_rmse_norm == val_rmse_norm else None
            model.train()  # Set back to train mode

        monitor_loss = val_loss if use_validation and val_loss is not None else epoch_loss
        scheduler.step(monitor_loss)

        monitor_rmse = (
            val_rmse_norm if use_validation and val_rmse_norm is not None else math.sqrt(epoch_loss)
        )
        monitor_rmse_mV = (
            val_rmse_mV if use_validation and val_rmse_mV is not None else train_rmse_mV
        )

        if monitor_rmse < best_rmse:
            best_rmse = monitor_rmse
            best_rmse_mV = monitor_rmse_mV
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_since_improve = 0
            history["best"] = {
                "epoch": best_epoch,
                "train_loss": epoch_loss,
                "train_rmse_mV": train_rmse_mV,
                "val_loss": val_loss,
                "val_rmse_mV": val_rmse_mV,
            }
            if verbose:
                print(f"✅ Best RMSE: {monitor_rmse_mV:.2f} mV at epoch {epoch + 1}")
            
            # Save checkpoint immediately when best model is found (to prevent data loss on interruption)
            try:
                checkpoint = {
                    "model_state_dict": best_state,
                    "training_info": {
                        "best_metric": "val_rmse_mV" if use_validation else "train_rmse_mV",
                        "best_rmse_mV": best_rmse_mV,
                        "best_rmse_norm": best_rmse,
                        "best_epoch": best_epoch,
                        "total_epochs": num_epochs,
                        "lr": lr,
                        "window_len": None,
                        "predict_delta": predict_delta,
                        "early_stop_patience": early_stop_patience,
                        "early_stop_window": early_stop_window,
                        "early_stop_delta_mV": early_stop_delta_mV,
                        "use_validation": use_validation,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "architecture_config": model.architecture_config,
                        "gru1_hidden": model.architecture_config["gru1_hidden"],
                        "gru2_hidden": model.architecture_config["gru2_hidden"],
                        "dense1_hidden": model.architecture_config["dense1_hidden"],
                        "dense2_hidden": model.architecture_config["dense2_hidden"],
                    },
                    "network_architecture": str(model),
                }
                best_model_path = f"best_model_gru_rmse{best_rmse_mV:.2f}mV.pth"
                # Delete previous checkpoint if it exists
                if previous_best_model_path is not None and os.path.exists(previous_best_model_path):
                    try:
                        os.remove(previous_best_model_path)
                        if verbose:
                            print(f"  Deleted previous checkpoint: {previous_best_model_path}")
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Failed to delete previous checkpoint: {e}")
                # Ensure directory exists
                checkpoint_dir = os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else "."
                if checkpoint_dir and not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir, exist_ok=True)
                
                torch.save(checkpoint, best_model_path)
                previous_best_model_path = best_model_path  # Update for next iteration
                history["best"]["checkpoint_path"] = best_model_path
                history["checkpoint_path"] = best_model_path
                if verbose:
                    print(f"  Checkpoint saved: {best_model_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to save checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            epochs_since_improve += 1

        epoch_time = time.time() - epoch_start_time
        if grad_norms_before and grad_norms_after:
            avg_grad_norm_before = float(np.mean(grad_norms_before))
            avg_grad_norm_after = float(np.mean(grad_norms_after))
            grad_str = f"Grad: {avg_grad_norm_before:.2f} → {avg_grad_norm_after:.2f}"
        else:
            grad_str = "Grad: N/A"
        if verbose:
            msg = (
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Loss: {epoch_loss:.4e} | Train RMSE: {train_rmse_mV:.2f} mV | "
                f"{grad_str} | "
                f"Time: {epoch_time:.1f}s"
            )
            if use_validation and val_loss is not None:
                msg += f" | Val Loss: {val_loss:.4e} | Val RMSE: {val_rmse_mV:.2f} mV"
            print(msg)

        if early_stop_patience and epochs_since_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping triggered (no improvement for {early_stop_patience} epochs).")
            break

        if early_stop_window > 0 and train_rmse_mV == train_rmse_mV:
            rmse_window.append(train_rmse_mV)
            if (
                rmse_window.maxlen == len(rmse_window)
                and (max(rmse_window) - min(rmse_window)) <= early_stop_delta_mV
            ):
                if verbose:
                    print(
                        f"Early stopping triggered (ΔRMSE <= {early_stop_delta_mV:.3f} mV over "
                        f"{early_stop_window} epochs)."
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # If no improvement, use the last epoch's model
        best_state = model.state_dict()
        if best_epoch == -1:
            best_epoch = num_epochs  # Use last epoch if no improvement

    if best_epoch == -1:
        best_epoch = history["best"].get("epoch", num_epochs) if history["best"] else num_epochs

    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "training_info": {
            "best_metric": "val_rmse_mV" if use_validation else "train_rmse_mV",
            "best_rmse_mV": best_rmse_mV,
            "best_rmse_norm": best_rmse,
            "best_epoch": best_epoch,
            "total_epochs": num_epochs,
            "lr": lr,
            "window_len": None,  # No longer used in pure RNN approach
            "predict_delta": predict_delta,
            "early_stop_patience": early_stop_patience,
            "early_stop_window": early_stop_window,
            "early_stop_delta_mV": early_stop_delta_mV,
            "use_validation": use_validation,
            "hidden_size": hidden_size,  # Kept for backward compatibility
            "num_layers": num_layers,  # Kept for backward compatibility
            # New architecture info
            "architecture_config": model.architecture_config,
            "gru1_hidden": model.architecture_config["gru1_hidden"],
            "gru2_hidden": model.architecture_config["gru2_hidden"],
            "dense1_hidden": model.architecture_config["dense1_hidden"],
            "dense2_hidden": model.architecture_config["dense2_hidden"],
        },
        "network_architecture": str(model),
    }
    best_model_path = f"best_model_gru_rmse{best_rmse_mV:.2f}mV.pth"
    # Delete previous checkpoint if it exists (only if it's different from the one we're about to save)
    if previous_best_model_path is not None and os.path.exists(previous_best_model_path) and previous_best_model_path != best_model_path:
        try:
            os.remove(previous_best_model_path)
            if verbose:
                print(f"Deleted previous checkpoint: {previous_best_model_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to delete previous checkpoint: {e}")
    
    # Ensure directory exists
    checkpoint_dir = os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else "."
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        torch.save(checkpoint, best_model_path)
        if verbose:
            print(f"Best model saved to: {best_model_path}")
    except Exception as e:
        if verbose:
            print(f"Error: Failed to save checkpoint: {e}")
            import traceback
            traceback.print_exc()

    history["best"]["checkpoint_path"] = best_model_path
    history["checkpoint_path"] = best_model_path

    return model, history


def train_gru_benchmark(
    train_dict_list: Sequence[Dict[str, Any]],
    val_dict_list: Optional[Sequence[Dict[str, Any]]] = None,
    num_epochs: int = 1000,
    lr: float = 5e-4,
    device: Union[str, torch.device] = "cpu",
    pretrained_model_path: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    predict_delta: bool = False,
    early_stop_patience: Optional[int] = None,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    verbose: bool = True,
    hidden_size: int = 12,
    num_layers: int = 1,
    gru1_hidden: int = 32,
    gru2_hidden: int = 24,
    dense1_hidden: int = 16,
    dense2_hidden: int = 8,
    loss_scale: float = 1.0,
    tbptt_length: Optional[int] = 300,
) -> Dict[str, Any]:
    """
    Batch size: Number of profiles to process simultaneously on GPU.
                If None, processes all profiles in one batch (GPU efficient).
    
    Early stopping:
        - If early_stop_patience is None, it will be set to early_stop_window value
        - This allows using a single early_stop_window parameter for both conditions
    """
    import sys
    sys.stdout.flush()  # Force flush output
    print("=" * 70, flush=True)
    print("[train_gru_benchmark] FUNCTION CALLED!", flush=True)
    print("=" * 70, flush=True)
    print(f"[train_gru_benchmark] Function called with {len(train_dict_list)} training profiles", flush=True)
    print(f"[train_gru_benchmark] batch_size={batch_size}, hidden_size={hidden_size}", flush=True)
    print(f"[train_gru_benchmark] num_epochs={num_epochs}, lr={lr}, device={device}", flush=True)
    
    # If early_stop_patience is not specified, use early_stop_window value
    if early_stop_patience is None:
        early_stop_patience = early_stop_window
        if verbose:
            print(f"[train_gru_benchmark] early_stop_patience not specified, using early_stop_window value: {early_stop_patience}", flush=True)
    
    try:
        model, history = train_battery_gru(
            train_dict_list=train_dict_list,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
            pretrained_model_path=pretrained_model_path,
            predict_delta=predict_delta,
            early_stop_patience=early_stop_patience,
            early_stop_window=early_stop_window,
            early_stop_delta_mV=early_stop_delta_mV,
            val_dict_list=val_dict_list,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_size=batch_size,
            gru1_hidden=gru1_hidden,
            gru2_hidden=gru2_hidden,
            dense1_hidden=dense1_hidden,
            dense2_hidden=dense2_hidden,
            loss_scale=loss_scale,
            tbptt_length=tbptt_length,
        )
        print(f"[train_gru_benchmark] Training completed successfully")
        return {"model": model, "history": history}
    except Exception as e:
        print(f"[train_gru_benchmark] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def simulate_battery_gru(
    model: BatteryGRUWrapper,
    data_dict: Dict[str, Any],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, np.ndarray]:
    model.eval()
    inputs, targets = _build_feature_tensor(data_dict, torch.device(device))
    with torch.no_grad():
        preds, _ = model(inputs)
    preds = preds.squeeze(0).cpu().numpy()
    targets = targets.squeeze(0).cpu().numpy()

    Y_mean = data_dict.get("Y_mean", 0.0)
    Y_std = data_dict.get("Y_std", 1.0)
    preds_denorm = preds * Y_std + Y_mean
    targets_denorm = targets * Y_std + Y_mean
    Vtotal_pred = np.array(data_dict["V_spme"]) + preds_denorm.squeeze(-1)
    Vtotal_meas = np.array(data_dict.get("V_meas", data_dict["V_ref"]))

    rmse_vcorr = float(np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2)) * 1000)
    rmse_vtotal = float(np.sqrt(np.mean((Vtotal_pred - Vtotal_meas) ** 2)) * 1000)

    return {
        "Vcorr_pred": preds_denorm.squeeze(-1),
        "Vcorr_target": targets_denorm.squeeze(-1),
        "Vtotal_pred": Vtotal_pred,
        "Vtotal_meas": Vtotal_meas,
        "rmse_vcorr_mV": rmse_vcorr,
        "rmse_vtotal_mV": rmse_vtotal,
    }


def _prepare_profile_tensors(profile: Dict[str, Any], device: torch.device):
    prepared = prepare_profile_for_gru(profile)
    feature_array = np.stack(
        [
            prepared["V_spme_norm"].astype(np.float32),
            prepared["ocv"].astype(np.float32),
            prepared["Y"].astype(np.float32),
            prepared["SOC"].astype(np.float32),
            prepared["I"].astype(np.float32),
            prepared["T"].astype(np.float32),
        ],
        axis=-1,
    )
    feature_tensor = torch.tensor(feature_array, dtype=torch.float32, device=device)
    y_norm = torch.tensor(prepared["Y"], dtype=torch.float32, device=device)
    return prepared, feature_tensor, y_norm


def _build_dynamic_window(
    feature_tensor: torch.Tensor,
    y_dynamic: List[torch.Tensor],
    start_idx: int,
    window_len: int,
) -> torch.Tensor:
    rows = []
    for offset in range(window_len):
        idx = start_idx + offset
        row = feature_tensor[idx].clone()
        row[2] = y_dynamic[idx]
        rows.append(row)
    return torch.stack(rows, dim=0).unsqueeze(0)


def _autoregressive_profile_train_pass_batch(
    model: BatteryGRUWrapper,
    profile_list: Sequence[Dict[str, Any]],
    predict_delta: bool,
    device: torch.device,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    loss_scale: float = 1.0,
    tbptt_length: Optional[int] = 300,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_norms_before: Optional[List[float]] = None,
    grad_norms_after: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Batch version: Process multiple profiles simultaneously on GPU.
    Each profile is auto-regressively trained (uses predictions, not ground truth).
    
    Returns: (loss_tensor, stats_dict)
    """
    # Ensure model is in training mode
    model.train()
    
    if not profile_list:
        return torch.tensor(0.0, device=device), {"num_steps": 0, "rmse_mV": float("nan")}
    
    # Prepare all profiles
    profile_data = []
    skipped_count = 0
    for profile in profile_list:
        prepared, feature_tensor, y_norm = _prepare_profile_tensors(profile, device)
        total_steps = y_norm.shape[0]
        # Pure RNN: need at least 2 timesteps (t=0 for initialization, t=1 for first prediction)
        if total_steps <= 1:
            skipped_count += 1
            continue
        profile_data.append({
            "prepared": prepared,
            "features": feature_tensor,  # [total_steps, 6]
            "y_norm": y_norm,  # [total_steps]
            "total_steps": total_steps,
        })
    
    if not profile_data:
        return {"num_steps": 0, "rmse_mV": float("nan"), "total_loss": 0.0, "valid_profiles": 0, "skipped_profiles": len(profile_list)}
    
    # Pad profiles to same length for batch processing
    max_length = max(p["total_steps"] for p in profile_data)
    batch_size_actual = len(profile_data)
    
    if verbose:
        if skipped_count > 0:
            print(f"    Skipped {skipped_count} profiles (length <= 1)")
        print(f"    Preparing batch: {batch_size_actual} profiles, max_length={max_length}")
    
    # Create batch tensors
    features_batch = torch.zeros(batch_size_actual, max_length, 6, device=device)
    y_norm_batch = torch.zeros(batch_size_actual, max_length, device=device)
    valid_mask = torch.zeros(batch_size_actual, max_length, dtype=torch.bool, device=device)
    y_std_list = []
    
    for i, p in enumerate(profile_data):
        valid_len = p["total_steps"]
        features_batch[i, :valid_len, :] = p["features"]
        y_norm_batch[i, :valid_len] = p["y_norm"]
        valid_mask[i, :valid_len] = True
        y_std_list.append(p["prepared"]["Y_std"])
    
    # Initialize predictions: t=0 uses ground truth
    pred_y_norm_batch = y_norm_batch.clone()
    
    # Pure RNN: process one timestep at a time, no sliding window
    losses: List[torch.Tensor] = []
    total_valid_steps = 0
    
    # Find max valid length
    valid_lengths = valid_mask.sum(dim=1)  # [batch_size]
    max_valid_length = valid_lengths.max().item()
    
    if verbose:
        print(f"    Starting RNN processing: {max_valid_length} timesteps (t=0 to t={max_valid_length-1})...")
    
    if max_valid_length <= 1:
        if verbose:
            print(f"    Skipping: max_length ({max_valid_length}) <= 1")
        return torch.tensor(0.0, device=device), {"num_steps": 0, "rmse_mV": float("nan")}
    
    # Process each timestep sequentially (pure RNN)
    import time
    rollout_start = time.time()
    
    # Initialize hidden states for all profiles in batch at t=0
    # Each profile has independent hidden state: [num_layers, batch_size, hidden_size]
    # gru1: bidirectional, so 2 directions per layer
    gru1_hidden = None  # Will be initialized to zero on first forward pass (t=0)
    gru2_hidden = None  # Will be initialized to zero on first forward pass (t=0)
    
    # TBPTT: Process in chunks if tbptt_length is specified
    if tbptt_length is not None and tbptt_length > 0:
        # Process in chunks: forward and backward in chunks, but hidden state carries over
        chunk_losses = []
        chunk_rmse_accum = []
        chunk_steps = 0
        
        for chunk_start in range(1, max_valid_length, tbptt_length):  # Start from t=1 (t=0 has no loss)
            chunk_end = min(chunk_start + tbptt_length, max_valid_length)
            chunk_losses_timestep = []
            
            # CRITICAL: Detach hidden state at the start of each chunk to break computation graph from previous chunk
            # This ensures each chunk has an independent computation graph for backward pass
            # We detach BEFORE the forward pass so that gradients don't flow across chunks
            if gru1_hidden is not None:
                gru1_hidden = gru1_hidden.detach()
                gru2_hidden = gru2_hidden.detach()
            
            # Forward pass for this chunk (hidden state continues from previous chunk, but detached at chunk start)
            # Within chunk, hidden state flows normally without detach to maintain sequence learning
            for t in range(chunk_start, chunk_end):
                # Check which profiles are valid at this timestep
                valid_at_t = t < valid_lengths
                if not valid_at_t.any():
                    continue
                
                # Get single timestep input: [batch_size, 1, 6]
                input_t = features_batch[:, t:t+1, :].clone()  # [batch_size, 1, 6]
                # TEST: Teacher forcing - use ground truth instead of prediction
                # Use previous ground truth value for input (teacher forcing)
                if t > 0:
                    input_t[:, 0, 2] = y_norm_batch[:, t - 1]  # Use ground truth Vcorr_true(k) instead of Vcorr_pred(k)
                # else: t == 0 uses ground truth from features_batch (no prediction available yet)
                
                # Forward pass with continuing hidden state
                # Hidden state is NOT detached within chunk to maintain sequence learning
                if gru1_hidden is None:
                    preds_seq, (gru1_hidden_new, gru2_hidden_new) = model(input_t, hidden=None)
                    gru1_hidden = gru1_hidden_new
                    gru2_hidden = gru2_hidden_new
                else:
                    preds_seq, (gru1_hidden_new, gru2_hidden_new) = model(input_t, hidden=(gru1_hidden, gru2_hidden))
                    # Update hidden state only for valid profiles
                    valid_mask_hidden = valid_at_t.float().unsqueeze(0).unsqueeze(-1)
                    gru1_hidden = gru1_hidden * (1 - valid_mask_hidden) + gru1_hidden_new * valid_mask_hidden
                    gru2_hidden = gru2_hidden * (1 - valid_mask_hidden) + gru2_hidden_new * valid_mask_hidden
                
                # Get prediction
                pred_raw = preds_seq[:, 0, 0]  # [batch_size]
                
                # Calculate loss
                # TEST: Model directly predicts Vcorr(k), not delta_Vcorr(k)
                # Use previous prediction value (detached) for input, but model output is Vcorr(k) directly
                prev_pred_val = pred_y_norm_batch[:, t - 1].detach() if t > 0 else pred_y_norm_batch[:, t - 1]
                
                # Model output is Vcorr_pred(k) directly (not delta)
                pred_value = pred_raw  # Direct prediction of Vcorr(k)
                target_value = y_norm_batch[:, t]  # Vcorr_true(k)
                loss_per_profile = (pred_value - target_value) ** 2
                
                # Only count valid profiles
                valid_loss = loss_per_profile * valid_at_t.float()
                timestep_loss = valid_loss.sum() / max(valid_at_t.sum().item(), 1)
                chunk_losses_timestep.append(timestep_loss)
                
                # Update predictions (auto-regressive) - detach to prevent graph explosion
                pred_y_norm_batch[:, t] = torch.where(valid_at_t, pred_value.detach(), pred_y_norm_batch[:, t])
                chunk_steps += valid_at_t.sum().item()
            
            # Backward pass for entire chunk (if optimizer is provided)
            # This is the correct TBPTT: backward chunk as a whole, not each timestep
            if chunk_losses_timestep and optimizer is not None:
                # Stack losses and compute mean for the chunk
                chunk_loss = torch.stack(chunk_losses_timestep).mean() * loss_scale
                
                optimizer.zero_grad()
                chunk_loss.backward()
                
                # Gradient clipping
                first_param = next(model.parameters())
                total_norm_sq = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq += param_norm ** 2
                total_norm = (total_norm_sq ** (1. / 2))
                
                if grad_norms_before is not None:
                    grad_norms_before.append(total_norm.item())
                
                # Manual gradient clipping
                max_norm = 1.0
                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(clip_coef)
                
                # Calculate gradient norm after clipping
                total_norm_sq_after = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq_after += param_norm ** 2
                grad_norm_after = (total_norm_sq_after ** (1. / 2)).item()
                
                if grad_norms_after is not None:
                    grad_norms_after.append(grad_norm_after)
                
                optimizer.step()
                
                # Detach hidden state AFTER backward to prevent gradient flow to next chunk
                if gru1_hidden is not None:
                    gru1_hidden = gru1_hidden.detach()
                    gru2_hidden = gru2_hidden.detach()
                
                chunk_losses.append(chunk_loss.item())
        
        # Return accumulated loss
        total_loss = sum(chunk_losses) if chunk_losses else 0.0
        
        # Calculate RMSE
        pred_slice = pred_y_norm_batch[:, 1:]
        target_slice = y_norm_batch[:, 1:]
        valid_slice = valid_mask[:, 1:]
        timestep_indices = torch.arange(1, max_length, device=device).unsqueeze(0)
        valid_timestep_mask = timestep_indices < valid_lengths.unsqueeze(1)
        valid_pred_mask = valid_slice & valid_timestep_mask
        
        if valid_pred_mask.any():
            pred_valid = pred_slice[valid_pred_mask]
            target_valid = target_slice[valid_pred_mask]
            rmse_norm = torch.sqrt(((pred_valid - target_valid) ** 2).mean()).item()
            avg_y_std = float(np.mean(y_std_list))
            rmse_mV = rmse_norm * avg_y_std * 1000
        else:
            rmse_mV = float("nan")
        
        return {
            "num_steps": chunk_steps,
            "rmse_mV": rmse_mV,
            "valid_profiles": batch_size_actual,
            "skipped_profiles": skipped_count,
            "tbptt_chunks": len(chunk_losses),
            "total_loss": total_loss,
        }
    
    # Original full BPTT (no TBPTT)
    # Process from t=0 to end (pure RNN: one timestep at a time)
    for t in range(max_valid_length):
        if verbose and t == 0:
            print(f"    Processing timesteps 0 to {max_valid_length - 1}...")
        # Check which profiles are valid at this timestep
        valid_at_t = t < valid_lengths
        if not valid_at_t.any():
            continue
        
        # Get single timestep input: [batch_size, 1, 6]
        # TEST: Teacher forcing - use ground truth instead of prediction
        input_t = features_batch[:, t:t+1, :].clone()  # [batch_size, 1, 6]
        input_t[:, 0, 2] = y_norm_batch[:, t]  # Use ground truth Vcorr_true(k) instead of Vcorr_pred(k) (teacher forcing)
        
        # Forward pass: one timestep at a time
        # Hidden state evolves from t=0 to profile end (never reset)
        if t == 0:
            # t=0: initialize hidden state to zero
            preds_seq, (gru1_hidden_new, gru2_hidden_new) = model(input_t, hidden=None)
            gru1_hidden = gru1_hidden_new
            gru2_hidden = gru2_hidden_new
        else:
            # t>0: use previous hidden state (continues evolving)
            preds_seq, (gru1_hidden_new, gru2_hidden_new) = model(input_t, hidden=(gru1_hidden, gru2_hidden))
            
            # Only update hidden state for valid profiles
            # For invalid profiles (ended), keep previous hidden state (frozen)
            # gru1_hidden: [num_layers*2, batch_size, hidden_size] (bidirectional, e.g., [2, batch_size, 32])
            # gru2_hidden: [num_layers, batch_size, hidden_size] (e.g., [1, batch_size, 24])
            
            # Create mask with correct shape for broadcasting
            valid_mask_hidden = valid_at_t.float().unsqueeze(0).unsqueeze(-1)  # [1, batch_size, 1]
            
            # Update hidden state only for valid profiles
            gru1_hidden = gru1_hidden * (1 - valid_mask_hidden) + gru1_hidden_new * valid_mask_hidden
            gru2_hidden = gru2_hidden * (1 - valid_mask_hidden) + gru2_hidden_new * valid_mask_hidden
        
        # Get prediction: [batch_size, 1, 1] -> [batch_size]
        pred_raw = preds_seq[:, 0, 0]  # [batch_size] - single timestep prediction
        
        # Calculate loss for this timestep (only for t >= 1, since we need previous value for delta)
        if t >= 1:
            # TEST: Model directly predicts Vcorr(k), not delta_Vcorr(k)
            # Model output is Vcorr_pred(k) directly (not delta)
            pred_value = pred_raw  # Direct prediction of Vcorr(k)
            target_value = y_norm_batch[:, t]  # Vcorr_true(k)
            loss_per_profile = (pred_value - target_value) ** 2
            
            # Only count valid profiles
            valid_loss = loss_per_profile * valid_at_t.float()
            losses.append(valid_loss.sum() / max(valid_at_t.sum().item(), 1))
            
            # Update predictions (auto-regressive: use prediction for next step)
            pred_y_norm_batch[:, t] = torch.where(valid_at_t, pred_value, pred_y_norm_batch[:, t])
            total_valid_steps += valid_at_t.sum().item()
        else:
            # t=0: no loss calculation (use ground truth, no prediction yet)
            # Keep ground truth for t=0
            pass
    
    if not losses:
        return {"num_steps": 0, "rmse_mV": float("nan"), "total_loss": 0.0, "valid_profiles": batch_size_actual, "skipped_profiles": skipped_count}
    
    # Calculate final loss and scale by loss_scale
    loss_tensor = torch.stack(losses).mean() * loss_scale
    
    # Backward pass (if optimizer is provided and TBPTT is not used)
    if optimizer is not None:
        optimizer.zero_grad()
        loss_tensor.backward()
        
        # Gradient clipping
        first_param = next(model.parameters())
        total_norm_sq = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm ** 2
        total_norm = (total_norm_sq ** (1. / 2))
        
        if grad_norms_before is not None:
            grad_norms_before.append(total_norm.item())
        
        # Manual gradient clipping
        max_norm = 1.0
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        # Calculate gradient norm after clipping
        total_norm_sq_after = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq_after += param_norm ** 2
        grad_norm_after = (total_norm_sq_after ** (1. / 2)).item()
        
        if grad_norms_after is not None:
            grad_norms_after.append(grad_norm_after)
        
        optimizer.step()
    
    # Calculate RMSE
    # Pure RNN: consider predictions from t=1 onwards (t=0 uses ground truth)
    pred_slice = pred_y_norm_batch[:, 1:]  # [batch_size, max_length - 1]
    target_slice = y_norm_batch[:, 1:]  # [batch_size, max_length - 1]
    valid_slice = valid_mask[:, 1:]  # [batch_size, max_length - 1]
    
    # Create mask: valid positions where timestep < valid_length for each profile
    timestep_indices = torch.arange(1, max_length, device=device).unsqueeze(0)  # [1, max_length - 1]
    valid_timestep_mask = timestep_indices < valid_lengths.unsqueeze(1)  # [batch_size, max_length - 1]
    valid_pred_mask = valid_slice & valid_timestep_mask
    
    if valid_pred_mask.any():
        pred_valid = pred_slice[valid_pred_mask]
        target_valid = target_slice[valid_pred_mask]
        rmse_norm = torch.sqrt(((pred_valid - target_valid) ** 2).mean()).item()
        avg_y_std = float(np.mean(y_std_list))
        rmse_mV = rmse_norm * avg_y_std * 1000
    else:
        rmse_mV = float("nan")
    
    return {
        "num_steps": total_valid_steps, 
        "rmse_mV": rmse_mV,
        "valid_profiles": batch_size_actual,
        "skipped_profiles": skipped_count,
        "total_loss": loss_tensor.item(),
    }


def _autoregressive_profile_train_pass(
    model: BatteryGRUWrapper,
    profile: Dict[str, Any],
    window_len: int,
    predict_delta: bool,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Single profile version (legacy, kept for backward compatibility).
    For batch processing, use _autoregressive_profile_train_pass_batch instead.
    """
    prepared, feature_tensor, y_norm = _prepare_profile_tensors(profile, device)
    total_steps = y_norm.shape[0]
    if total_steps <= window_len:
        return None

    y_dynamic: List[torch.Tensor] = [y_norm[i].clone() for i in range(total_steps)]
    losses: List[torch.Tensor] = []
    sum_sq_norm = 0.0
    num_steps = 0

    for t in range(window_len, total_steps):
        window_tensor = _build_dynamic_window(feature_tensor, y_dynamic, t - window_len, window_len)
        preds_seq, _ = model(window_tensor)
        pred_raw = preds_seq[:, -1, 0]

        if predict_delta:
            target_delta = y_norm[t] - y_dynamic[t - 1]
            loss = (pred_raw - target_delta) ** 2
            pred_value = y_dynamic[t - 1] + pred_raw
        else:
            target_value = y_norm[t]
            loss = (pred_raw - target_value) ** 2
            pred_value = pred_raw

        losses.append(loss.mean())
        err = (pred_value.squeeze(0) - y_norm[t]).detach().cpu().item()
        sum_sq_norm += err * err
        y_dynamic[t] = pred_value.squeeze(0)
        num_steps += 1

    if num_steps == 0:
        return None

    loss_tensor = torch.stack(losses).mean() * 1e6  # Legacy function - using fixed 1e6 scale
    rmse_norm = math.sqrt(max(sum_sq_norm / num_steps, 1e-12))
    rmse_mV = rmse_norm * prepared["Y_std"] * 1000
    return loss_tensor, {"num_steps": num_steps, "rmse_mV": rmse_mV}


def _evaluate_gru_profiles(
    model: BatteryGRUWrapper,
    dict_list: Sequence[Dict[str, Any]],
    predict_delta: bool,
    device: torch.device,
) -> Dict[str, float]:
    if not dict_list:
        return {
            "avg_rmse_vcorr_mV": float("nan"),
            "avg_rmse_norm": float("nan"),
            "avg_Y_std": float("nan"),
        }

    rmse_norm_list: List[float] = []
    rmse_vcorr_list: List[float] = []
    y_std_list: List[float] = []

    model.eval()
    with torch.no_grad():
        for profile in dict_list:
            prepared = prepare_profile_for_gru(profile)
            feature_array = np.stack(
                [
                    prepared["V_spme_norm"].astype(np.float32),
                    prepared["ocv"].astype(np.float32),
                    prepared["Y"].astype(np.float32),
                    prepared["SOC"].astype(np.float32),
                    prepared["I"].astype(np.float32),
                    prepared["T"].astype(np.float32),
                ],
                axis=-1,
            )
            preds_norm = _gru_autoregressive_rollout(
                model=model,
                feature_array=feature_array,
                initial_y=prepared["Y"],
                predict_delta=predict_delta,
            )
            y_norm = np.asarray(prepared["Y"], dtype=np.float32)
            # Pure RNN: use all timesteps (t=0 is ground truth, t>=1 are predictions)
            mask = np.ones_like(y_norm, dtype=bool)
            mask[0] = False  # Skip t=0 (ground truth, no prediction)
            mask &= np.isfinite(preds_norm) & np.isfinite(y_norm)
            if not np.any(mask):
                continue
            rmse_norm = math.sqrt(float(np.mean((preds_norm[mask] - y_norm[mask]) ** 2)))
            rmse_vcorr = rmse_norm * prepared["Y_std"] * 1000
            rmse_norm_list.append(rmse_norm)
            rmse_vcorr_list.append(rmse_vcorr)
            y_std_list.append(prepared["Y_std"])

    return {
        "avg_rmse_vcorr_mV": float(np.nanmean(rmse_vcorr_list)) if rmse_vcorr_list else float("nan"),
        "avg_rmse_norm": float(np.nanmean(rmse_norm_list)) if rmse_norm_list else float("nan"),
        "avg_Y_std": float(np.nanmean(y_std_list)) if y_std_list else float("nan"),
    }


def _resolve_profile_name(profile: Dict[str, Any], fallback: str) -> str:
    for key in ("name", "profile_name", "id", "label"):
        if key in profile:
            return str(profile[key])
    return fallback


def _gru_autoregressive_rollout(
    model: BatteryGRUWrapper,
    feature_array: np.ndarray,
    initial_y: np.ndarray,
    predict_delta: bool,
) -> np.ndarray:
    """
    Pure RNN inference: process one timestep at a time.
    Hidden state evolves from t=0 to profile end (never reset).
    """
    features_autoreg = feature_array.copy()
    total_steps = feature_array.shape[0]
    preds_norm = np.full(total_steps, np.nan, dtype=np.float32)
    
    # t=0: use ground truth (no prediction)
    preds_norm[0] = initial_y[0]

    model.eval()
    gru1_hidden = None
    gru2_hidden = None
    
    # Pure RNN: process from t=0 to end, one timestep at a time
    for idx in range(total_steps):
        if idx == 0:
            # t=0: initialize hidden state to zero, use ground truth
            input_t = torch.from_numpy(features_autoreg[idx:idx+1]).unsqueeze(0).to(model.device, dtype=torch.float32)  # [1, 1, 6]
            with torch.no_grad():
                preds_seq, (gru1_hidden, gru2_hidden) = model(input_t, hidden=None)
            # t=0: no prediction, use ground truth (already set in preds_norm[0] = initial_y[0])
            continue
        
        # t>=1: process one timestep at a time with hidden state continuation
        input_t = torch.from_numpy(features_autoreg[idx:idx+1]).unsqueeze(0).to(model.device, dtype=torch.float32)  # [1, 1, 6]
        # Use predicted Vcorr (auto-regressive)
        # preds_norm[idx - 1] is already the predicted Vcorr value (not delta) from previous timestep
        input_t[0, 0, 2] = torch.tensor(float(preds_norm[idx - 1]), dtype=torch.float32, device=model.device)
        
        with torch.no_grad():
            preds_seq, (gru1_hidden, gru2_hidden) = model(input_t, hidden=(gru1_hidden, gru2_hidden))
            next_pred = preds_seq[0, 0, 0].item()

        # TEST: Model directly predicts Vcorr(k), not delta_Vcorr(k)
        # No need to add previous value - model output is already Vcorr(k)
        # if predict_delta:
        #     prev_val = preds_norm[idx - 1]
        #     next_pred = prev_val + next_pred

        preds_norm[idx] = next_pred
        features_autoreg[idx, 2] = next_pred

    return preds_norm


def run_gru_benchmark_inference(
    dict_list: Sequence[Dict[str, Any]],
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    predict_delta: Optional[bool] = None,
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Auto-regressive rollout (teacher-forcing warmup, predictive feedback) over a list of profiles.
    """
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_info = checkpoint.get("training_info", {})

    # window_len is no longer used in pure RNN approach
    # Keep for backward compatibility with old checkpoints, but don't use it
    inferred_window = training_info.get("window_len", None)
    inferred_predict_delta = training_info.get("predict_delta", predict_delta)
    if inferred_predict_delta is None:
        inferred_predict_delta = False

    # Check checkpoint structure by examining state_dict keys
    state_dict = checkpoint["model_state_dict"]
    has_gru1 = any("gru1" in key for key in state_dict.keys())
    has_gru = any(key.startswith("gru.") and "gru1" not in key and "gru2" not in key for key in state_dict.keys())
    
    # Get model architecture from checkpoint
    architecture_config = training_info.get("architecture_config", {})
    
    # For backward compatibility
    inferred_hidden_size = training_info.get("hidden_size", 12)
    inferred_num_layers = training_info.get("num_layers", 1)
    
    if has_gru1:
        # New architecture: Unidirectional GRU1 + Unidirectional GRU2
        gru1_hidden = architecture_config.get("gru1_hidden", training_info.get("gru1_hidden", 32))
        gru2_hidden = architecture_config.get("gru2_hidden", training_info.get("gru2_hidden", 24))
        dense1_hidden = architecture_config.get("dense1_hidden", training_info.get("dense1_hidden", 16))
        dense2_hidden = architecture_config.get("dense2_hidden", training_info.get("dense2_hidden", 8))
        if verbose:
            print(f"[run_gru_benchmark_inference] Loading new architecture: GRU1({gru1_hidden}) → GRU2({gru2_hidden}) → Dense({dense1_hidden}) → Dense({dense2_hidden}) → Linear(1)")
        
        model = BatteryGRUWrapper(
            input_size=6,
            hidden_size=inferred_hidden_size,
            num_layers=inferred_num_layers,
            device=device,
            gru1_hidden=gru1_hidden,
            gru2_hidden=gru2_hidden,
            dense1_hidden=dense1_hidden,
            dense2_hidden=dense2_hidden,
        ).to(device)
        model.load_state_dict(state_dict)
    elif has_gru:
        # Old architecture: Single Unidirectional GRU (remap to gru2)
        # Infer gru_hidden from state_dict
        if "gru.weight_ih_l0" in state_dict:
            gru_hidden_from_state = state_dict["gru.weight_ih_l0"].shape[0] // 3  # GRU has 3 gates
            gru_hidden = architecture_config.get("gru_hidden", training_info.get("gru_hidden", gru_hidden_from_state))
        else:
            gru_hidden = architecture_config.get("gru_hidden", training_info.get("gru_hidden", 24))
        
        dense1_hidden = architecture_config.get("dense1_hidden", training_info.get("dense1_hidden", 16))
        dense2_hidden = architecture_config.get("dense2_hidden", training_info.get("dense2_hidden", 8))
        
        if verbose:
            print(f"[run_gru_benchmark_inference] Loading old architecture (single GRU): GRU({gru_hidden}) → Dense({dense1_hidden}) → Dense({dense2_hidden}) → Linear(1)")
            print(f"[run_gru_benchmark_inference] Remapping 'gru' → 'gru2' (skipping gru1)")
        
        # Create model with new structure, but remap old "gru" weights to "gru2"
        # For old architecture, gru2 should receive input_size (6) instead of gru1 output (64)
        model = BatteryGRUWrapper(
            input_size=6,
            hidden_size=inferred_hidden_size,
            num_layers=inferred_num_layers,
            device=device,
            gru1_hidden=32,  # Will be ignored (not loaded)
            gru2_hidden=gru_hidden,  # This will receive the old "gru" weights
            dense1_hidden=dense1_hidden,
            dense2_hidden=dense2_hidden,
        ).to(device)
        
        # Override gru2 input size to match old checkpoint (input_size=6 instead of 64)
        model.gru2 = nn.GRU(
            input_size=6,  # Old checkpoint uses input_size directly
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        ).to(device)
        
        # Remap old "gru" weights to "gru2"
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("gru."):
                new_key = key.replace("gru.", "gru2.", 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        # Fallback to defaults
        if verbose:
            print(f"[run_gru_benchmark_inference] Could not detect architecture, using defaults")
        gru1_hidden = architecture_config.get("gru1_hidden", training_info.get("gru1_hidden", 32))
        gru2_hidden = architecture_config.get("gru2_hidden", training_info.get("gru2_hidden", 24))
        dense1_hidden = architecture_config.get("dense1_hidden", training_info.get("dense1_hidden", 16))
        dense2_hidden = architecture_config.get("dense2_hidden", training_info.get("dense2_hidden", 8))
        
        model = BatteryGRUWrapper(
            input_size=6,
            hidden_size=inferred_hidden_size,
            num_layers=inferred_num_layers,
            device=device,
            gru1_hidden=gru1_hidden,
            gru2_hidden=gru2_hidden,
            dense1_hidden=dense1_hidden,
            dense2_hidden=dense2_hidden,
        ).to(device)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    results: List[Dict[str, Any]] = []
    rmse_vcorr_list: List[float] = []
    rmse_vtotal_list: List[float] = []
    y_std_list: List[float] = []

    total_profiles = len(dict_list)
    if verbose:
        print(f"\n[GRU TEST] Loading checkpoint: {checkpoint_path}")
        if training_info:
            print("  --- Checkpoint Metadata ---")
            for key in sorted(training_info.keys()):
                print(f"  {key}: {training_info[key]}")
        else:
            print("  (no training_info metadata found)")
        print(f"[GRU TEST] Total profiles to evaluate: {total_profiles}")
        print("=" * 60)

    for idx, profile in enumerate(dict_list):
        prepared = prepare_profile_for_gru(profile)
        y_norm = np.asarray(prepared["Y"], dtype=np.float32)
        feature_array = np.stack(
            [
                prepared["V_spme_norm"].astype(np.float32),
                prepared["ocv"].astype(np.float32),
                y_norm,
                prepared["SOC"].astype(np.float32),
                prepared["I"].astype(np.float32),
                prepared["T"].astype(np.float32),
            ],
            axis=-1,
        )
        total_steps = feature_array.shape[0]
        # Pure RNN: need at least 2 timesteps (t=0 for initialization, t=1 for first prediction)
        if total_steps <= 1:
            if verbose:
                print(
                    f"[GRU TEST] Skipping profile {idx} "
                    f"(length {total_steps} <= 1)"
                )
            continue

        preds_norm = _gru_autoregressive_rollout(
            model=model,
            feature_array=feature_array,
            initial_y=y_norm,
            predict_delta=inferred_predict_delta,
        )

        y_mean = prepared["Y_mean"]
        y_std = prepared["Y_std"]
        pred_vcorr = preds_norm * y_std + y_mean
        target_vcorr = y_norm * y_std + y_mean
        v_spme = prepared["V_spme"]
        v_meas = prepared["V_meas"]
        time_array = prepared["time"]
        soc_array = prepared["SOC"]

        # Pure RNN: use all timesteps (t=0 is ground truth, t>=1 are predictions)
        mask_valid = np.ones(total_steps, dtype=bool)
        mask_valid[0] = False  # Skip t=0 (ground truth, no prediction)
        mask_valid &= np.isfinite(pred_vcorr) & np.isfinite(target_vcorr)

        if np.any(mask_valid):
            rmse_vcorr = float(
                np.sqrt(np.mean((pred_vcorr[mask_valid] - target_vcorr[mask_valid]) ** 2)) * 1000
            )
        else:
            rmse_vcorr = float("nan")

        if v_meas is not None and v_spme is not None:
            vtotal_pred = v_spme + pred_vcorr
            vtotal_target = v_meas
            mask_vtotal = mask_valid & np.isfinite(vtotal_target)
            if np.any(mask_vtotal):
                rmse_vtotal = float(
                    np.sqrt(np.mean((vtotal_pred[mask_vtotal] - vtotal_target[mask_vtotal]) ** 2))
                    * 1000
                )
            else:
                rmse_vtotal = float("nan")
        else:
            vtotal_pred = None
            vtotal_target = None
            rmse_vtotal = float("nan")

        rmse_vcorr_list.append(rmse_vcorr)
        rmse_vtotal_list.append(rmse_vtotal)
        y_std_list.append(prepared["Y_std"])

        results.append(
            {
                "profile_index": idx,
                "profile_name": _resolve_profile_name(profile, f"profile_{idx}"),
                "time": time_array,
                "SOC": soc_array,
                "V_meas": v_meas,
                "V_spme": v_spme,
                "Vcorr_pred": pred_vcorr,
                "Vcorr_target": target_vcorr,
                "Vtotal_pred": vtotal_pred,
                "Vtotal_target": vtotal_target,
                "rmse_vcorr_mV": rmse_vcorr,
                "rmse_vtotal_mV": rmse_vtotal,
                "rmse": rmse_vtotal / 1000.0 if np.isfinite(rmse_vtotal) else float("nan"),
                "rmse_mV": rmse_vtotal,
            }
        )

    metrics = {
        "num_profiles": len(results),
        "avg_rmse_vcorr_mV": float(np.nanmean(rmse_vcorr_list)) if rmse_vcorr_list else float("nan"),
        "median_rmse_vcorr_mV": float(np.nanmedian(rmse_vcorr_list))
        if rmse_vcorr_list
        else float("nan"),
        "avg_rmse_vtotal_mV": float(np.nanmean(rmse_vtotal_list))
        if rmse_vtotal_list
        else float("nan"),
        "median_rmse_vtotal_mV": float(np.nanmedian(rmse_vtotal_list))
        if rmse_vtotal_list
        else float("nan"),
        "checkpoint_training_info": training_info,
        "window_len_used": None,  # No longer used in pure RNN approach
        "predict_delta_used": inferred_predict_delta,
        "checkpoint_path": str(checkpoint_path),
        "avg_Y_std": float(np.nanmean(y_std_list)) if y_std_list else float("nan"),
    }

    if verbose:
        print("[GRU TEST] Completed evaluation.")
        print(
            f"[GRU TEST] RMSE Vcorr -> avg: {metrics['avg_rmse_vcorr_mV']:.2f} mV, "
            f"median: {metrics['median_rmse_vcorr_mV']:.2f} mV"
        )
        print(
            f"[GRU TEST] RMSE Vtotal -> avg: {metrics['avg_rmse_vtotal_mV']:.2f} mV, "
            f"median: {metrics['median_rmse_vtotal_mV']:.2f} mV"
        )

    return results, metrics

