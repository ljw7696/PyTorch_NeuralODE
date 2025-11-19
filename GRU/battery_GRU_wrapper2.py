import copy
import math
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
    ):
        super().__init__()
        self.device = torch.device(device)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

        self.apply(self._init_weights)
        self.hidden_state: Optional[torch.Tensor] = None

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
        num_layers = self.gru.num_layers
        hidden_size = self.gru.hidden_size
        self.hidden_state = torch.zeros(
            num_layers, batch_size, hidden_size, device=self.device
        )

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tensor [batch, seq_len, input_size]
            hidden: Optional initial hidden state
        Returns:
            preds: Tensor [batch, seq_len, 1]
            hidden: Final hidden state
        """
        if hidden is None:
            hidden = torch.zeros(
                self.gru.num_layers,
                inputs.size(0),
                self.gru.hidden_size,
                device=inputs.device,
            )

        outputs, hidden = self.gru(inputs, hidden)
        preds = self.head(outputs)
        return preds, hidden


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
    window_len: int = 50,
    predict_delta: bool = True,
    early_stop_patience: int = 50,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    val_dict_list: Optional[Sequence[Dict[str, Any]]] = None,
    shuffle_profiles: bool = True,
    hidden_size: int = 12,
    num_layers: int = 1,
    batch_size: Optional[int] = None,
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
    model = BatteryGRUWrapper(input_size=6, hidden_size=hidden_size, num_layers=num_layers, device=device).to(device)
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
        print(f"Architecture : GRU (input=6, hidden={model.gru.hidden_size}, layers={model.gru.num_layers})", flush=True)
        print(f"Head         : Linear({model.gru.hidden_size} -> 1)", flush=True)
        print(f"Device       : {device}", flush=True)
        print(f"Epochs       : {num_epochs}", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"Window len   : {window_len}", flush=True)
        print(f"Predict delta: {predict_delta}", flush=True)
        print(f"Batch size   : {batch_size if batch_size else 'all profiles'}", flush=True)
        print(f"Train profiles: {len(train_dict_list)}", flush=True)
        print(f"Val profiles  : {len(val_dict_list) if val_dict_list else 0}", flush=True)
        print(f"Early-stop    : patience={early_stop_patience}, window={early_stop_window}, Δ={early_stop_delta_mV} mV", flush=True)
        print("=" * 70 + "\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-3, threshold_mode="rel"
    )

    history: Dict[str, Dict[str, Optional[float]]] = {"best": {}}
    best_rmse = float("inf")
    best_rmse_mV = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    rmse_window = deque(maxlen=early_stop_window)
    use_validation = val_dict_list is not None and len(val_dict_list) > 0

    model.train()
    import time
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        if shuffle_profiles:
            random.shuffle(train_dict_list)  # type: ignore[arg-type]

        epoch_loss = 0.0
        epoch_steps = 0
        train_rmse_mV_accum: List[float] = []

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
            loss_batch, stats = _autoregressive_profile_train_pass_batch(
                model=model,
                profile_list=profile_batch,
                window_len=window_len,
                predict_delta=predict_delta,
                device=device,
                batch_size=batch_size,
                verbose=verbose and epoch == 0 and batch_idx == 0,
            )
            
            if stats["num_steps"] == 0:
                continue

            # Track valid/skipped profiles
            total_valid_profiles_epoch += stats.get("valid_profiles", 0)
            total_skipped_profiles_epoch += stats.get("skipped_profiles", 0)

            optimizer.zero_grad()
            loss_batch.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            optimizer.step()

            epoch_loss += loss_batch.item() * stats["num_steps"]
            epoch_steps += stats["num_steps"]
            train_rmse_mV_accum.append(stats["rmse_mV"])
        
        if verbose and epoch == 0:
            print(f"  [Epoch {epoch + 1}] Valid profiles: {total_valid_profiles_epoch}, Skipped: {total_skipped_profiles_epoch} (length <= {window_len})")

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
                window_len=window_len,
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
        else:
            epochs_since_improve += 1

        epoch_time = time.time() - epoch_start_time
        if verbose:
            msg = (
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Loss: {epoch_loss:.4e} | Train RMSE: {train_rmse_mV:.2f} mV | "
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

    if best_epoch == -1:
        best_epoch = history["best"].get("epoch", 0) if history["best"] else 0

    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "training_info": {
            "best_metric": "val_rmse_mV" if use_validation else "train_rmse_mV",
            "best_rmse_mV": best_rmse_mV,
            "best_rmse_norm": best_rmse,
            "best_epoch": best_epoch,
            "total_epochs": num_epochs,
            "lr": lr,
            "window_len": window_len,
            "predict_delta": predict_delta,
            "early_stop_patience": early_stop_patience,
            "early_stop_window": early_stop_window,
            "early_stop_delta_mV": early_stop_delta_mV,
            "use_validation": use_validation,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
        "network_architecture": str(model),
    }
    best_model_path = f"best_model_gru_rmse{best_rmse_mV:.2f}mV.pth"
    torch.save(checkpoint, best_model_path)
    if verbose:
        print(f"Best model saved to: {best_model_path}")

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
    window_len: int = 120,
    batch_size: Optional[int] = None,
    predict_delta: bool = False,
    early_stop_patience: int = 50,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    verbose: bool = True,
    hidden_size: int = 12,
    num_layers: int = 1,
) -> Dict[str, Any]:
    """
    Batch size: Number of profiles to process simultaneously on GPU.
                If None, processes all profiles in one batch (GPU efficient).
    """
    import sys
    sys.stdout.flush()  # Force flush output
    print("=" * 70, flush=True)
    print("[train_gru_benchmark] FUNCTION CALLED!", flush=True)
    print("=" * 70, flush=True)
    print(f"[train_gru_benchmark] Function called with {len(train_dict_list)} training profiles", flush=True)
    print(f"[train_gru_benchmark] batch_size={batch_size}, window_len={window_len}, hidden_size={hidden_size}", flush=True)
    print(f"[train_gru_benchmark] num_epochs={num_epochs}, lr={lr}, device={device}", flush=True)
    
    try:
        model, history = train_battery_gru(
            train_dict_list=train_dict_list,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
            pretrained_model_path=pretrained_model_path,
            window_len=window_len,
            predict_delta=predict_delta,
            early_stop_patience=early_stop_patience,
            early_stop_window=early_stop_window,
            early_stop_delta_mV=early_stop_delta_mV,
            val_dict_list=val_dict_list,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_size=batch_size,
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
    window_len: int,
    predict_delta: bool,
    device: torch.device,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        if total_steps <= window_len:
            skipped_count += 1
            continue
        profile_data.append({
            "prepared": prepared,
            "features": feature_tensor,  # [total_steps, 6]
            "y_norm": y_norm,  # [total_steps]
            "total_steps": total_steps,
        })
    
    if not profile_data:
        return torch.tensor(0.0, device=device), {"num_steps": 0, "rmse_mV": float("nan")}
    
    # Pad profiles to same length for batch processing
    max_length = max(p["total_steps"] for p in profile_data)
    batch_size_actual = len(profile_data)
    
    if verbose:
        if skipped_count > 0:
            print(f"    Skipped {skipped_count} profiles (length <= window_len={window_len})")
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
    
    # Initialize predictions with ground truth for initial window
    pred_y_norm_batch = y_norm_batch.clone()
    
    # Auto-regressive rollout: use predictions (not ground truth)
    losses: List[torch.Tensor] = []
    total_valid_steps = 0
    
    # Find max valid length
    valid_lengths = valid_mask.sum(dim=1)  # [batch_size]
    max_valid_length = valid_lengths.max().item()
    
    if verbose:
        print(f"    Starting auto-regressive rollout: {max_valid_length - window_len} timesteps...")
    
    if max_valid_length <= window_len:
        if verbose:
            print(f"    Skipping: max_length ({max_valid_length}) <= window_len ({window_len})")
        return torch.tensor(0.0, device=device), {"num_steps": 0, "rmse_mV": float("nan")}
    
    # Process each timestep auto-regressively
    import time
    rollout_start = time.time()
    for t in range(window_len, max_valid_length):
        if verbose and t == window_len:
            print(f"    Processing timesteps {window_len} to {max_valid_length - 1}...")
        # Check which profiles are valid at this timestep
        valid_at_t = (t < valid_lengths) & ((t - window_len) >= 0)
        if not valid_at_t.any():
            continue
        
        # Build windows: [batch_size, window_len, 6]
        # Use predictions (auto-regressive), not ground truth
        windows_batch = features_batch[:, t - window_len : t, :].clone()  # [batch_size, window_len, 6]
        windows_batch[:, :, 2] = pred_y_norm_batch[:, t - window_len : t]  # Use predictions
        
        # Predict next step
        preds_seq, _ = model(windows_batch)  # [batch_size, window_len, 1]
        pred_raw = preds_seq[:, -1, 0]  # [batch_size] - last timestep prediction
        
        # Calculate loss and update predictions
        if predict_delta:
            target_delta = y_norm_batch[:, t] - pred_y_norm_batch[:, t - 1]
            loss_per_profile = (pred_raw - target_delta) ** 2
            pred_value = pred_y_norm_batch[:, t - 1] + pred_raw
        else:
            target_value = y_norm_batch[:, t]
            loss_per_profile = (pred_raw - target_value) ** 2
            pred_value = pred_raw
        
        # Only count valid profiles
        valid_loss = loss_per_profile * valid_at_t.float()
        losses.append(valid_loss.sum() / max(valid_at_t.sum().item(), 1))
        
        # Update predictions (auto-regressive: use prediction for next step)
        pred_y_norm_batch[:, t] = torch.where(valid_at_t, pred_value, pred_y_norm_batch[:, t])
        total_valid_steps += valid_at_t.sum().item()
    
    if not losses:
        return torch.tensor(0.0, device=device), {"num_steps": 0, "rmse_mV": float("nan")}
    
    # Calculate final loss
    loss_tensor = torch.stack(losses).mean()
    
    # Calculate RMSE
    # Only consider predictions after window_len
    pred_slice = pred_y_norm_batch[:, window_len:]  # [batch_size, max_length - window_len]
    target_slice = y_norm_batch[:, window_len:]  # [batch_size, max_length - window_len]
    valid_slice = valid_mask[:, window_len:]  # [batch_size, max_length - window_len]
    
    # Create mask: valid positions where timestep < valid_length for each profile
    timestep_indices = torch.arange(window_len, max_length, device=device).unsqueeze(0)  # [1, max_length - window_len]
    valid_timestep_mask = timestep_indices < valid_lengths.unsqueeze(1)  # [batch_size, max_length - window_len]
    valid_pred_mask = valid_slice & valid_timestep_mask
    
    if valid_pred_mask.any():
        pred_valid = pred_slice[valid_pred_mask]
        target_valid = target_slice[valid_pred_mask]
        rmse_norm = torch.sqrt(((pred_valid - target_valid) ** 2).mean()).item()
        avg_y_std = float(np.mean(y_std_list))
        rmse_mV = rmse_norm * avg_y_std * 1000
    else:
        rmse_mV = float("nan")
    
    return loss_tensor, {
        "num_steps": total_valid_steps, 
        "rmse_mV": rmse_mV,
        "valid_profiles": batch_size_actual,
        "skipped_profiles": skipped_count,
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

    loss_tensor = torch.stack(losses).mean()
    rmse_norm = math.sqrt(max(sum_sq_norm / num_steps, 1e-12))
    rmse_mV = rmse_norm * prepared["Y_std"] * 1000
    return loss_tensor, {"num_steps": num_steps, "rmse_mV": rmse_mV}


def _evaluate_gru_profiles(
    model: BatteryGRUWrapper,
    dict_list: Sequence[Dict[str, Any]],
    window_len: int,
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
                window_len=window_len,
                predict_delta=predict_delta,
            )
            y_norm = np.asarray(prepared["Y"], dtype=np.float32)
            mask = np.ones_like(y_norm, dtype=bool)
            mask[:window_len] = False
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
    window_len: int,
    predict_delta: bool,
) -> np.ndarray:
    features_autoreg = feature_array.copy()
    total_steps = feature_array.shape[0]
    preds_norm = np.full(total_steps, np.nan, dtype=np.float32)
    warmup = min(window_len, total_steps)
    preds_norm[:warmup] = initial_y[:warmup]

    model.eval()
    for idx in range(window_len, total_steps):
        window_np = features_autoreg[idx - window_len : idx]
        window_tensor = (
            torch.from_numpy(window_np).unsqueeze(0).to(model.device, dtype=torch.float32)
        )
        with torch.no_grad():
            preds_seq, _ = model(window_tensor)
            next_pred = preds_seq[:, -1, 0].item()

        if predict_delta:
            prev_val = features_autoreg[idx - 1, 2]
            next_pred = prev_val + next_pred

        preds_norm[idx] = next_pred
        features_autoreg[idx, 2] = next_pred

    return preds_norm


def run_gru_benchmark_inference(
    dict_list: Sequence[Dict[str, Any]],
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    window_len: Optional[int] = None,
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

    inferred_window = training_info.get("window_len", window_len)
    if inferred_window is None:
        raise ValueError("window_len must be provided when checkpoint lacks metadata.")
    inferred_predict_delta = training_info.get("predict_delta", predict_delta)
    if inferred_predict_delta is None:
        inferred_predict_delta = False
    
    # Get model architecture from checkpoint
    inferred_hidden_size = training_info.get("hidden_size", hidden_size)
    if inferred_hidden_size is None:
        inferred_hidden_size = 12  # Default
    inferred_num_layers = training_info.get("num_layers", num_layers)
    if inferred_num_layers is None:
        inferred_num_layers = 1  # Default

    model = BatteryGRUWrapper(
        input_size=6,
        hidden_size=inferred_hidden_size,
        num_layers=inferred_num_layers,
        device=device
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
        if total_steps <= inferred_window:
            if verbose:
                print(
                    f"[GRU TEST] Skipping profile {idx} "
                    f"(length {total_steps} <= window {inferred_window})"
                )
            continue

        preds_norm = _gru_autoregressive_rollout(
            model=model,
            feature_array=feature_array,
            initial_y=y_norm,
            window_len=inferred_window,
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

        mask_valid = np.ones(total_steps, dtype=bool)
        mask_valid[:inferred_window] = False
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
        "window_len_used": inferred_window,
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

