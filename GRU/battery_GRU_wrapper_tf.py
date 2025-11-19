import copy
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from GRU.battery_GRU_wrapper import prepare_profile_for_gru


class BatteryGRUWrapperTF(nn.Module):
    """
    GRU model wrapper for teacher forcing training.
    Input: [V_spme_norm, ocv, Y, SOC, I, T] (6 features)
    Output: delta_Vcorr prediction (normalized)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 6,
        num_layers: int = 1,
        dropout: float = 0.0,
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        preds = self.head(outputs)
        return preds


def _prepare_profile_tensors_tf(
    profile: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
    """Prepare profile data for teacher forcing training."""
    prepared = prepare_profile_for_gru(profile)
    
    feature_array = np.stack(
        [
            prepared["V_spme_norm"],
            prepared["ocv"],
            prepared["Y"],
            prepared["SOC"],
            prepared["I"],
            prepared["T"],
        ],
        axis=-1,
    )
    
    feature_tensor = torch.from_numpy(feature_array).to(device, dtype=torch.float32)  # [T, 6]
    y_norm = torch.from_numpy(prepared["Y"]).to(device, dtype=torch.float32)  # [T]
    valid_length = len(feature_array)
    
    return feature_tensor, y_norm, valid_length, prepared["Y_mean"], prepared["Y_std"]


def _gru_autoregressive_rollout_tf(
    model: BatteryGRUWrapperTF,
    feature_array: np.ndarray,
    initial_y: np.ndarray,
    window_len: int,
    predict_delta: bool,
    device: torch.device,
) -> np.ndarray:
    """Autoregressive rollout for inference (uses predictions, not ground truth)."""
    features_autoreg = feature_array.copy()
    total_steps = feature_array.shape[0]
    preds_norm = np.full(total_steps, np.nan, dtype=np.float32)
    warmup = min(window_len, total_steps)
    preds_norm[:warmup] = initial_y[:warmup]

    model.eval()
    with torch.no_grad():
        for idx in range(window_len, total_steps):
            window_np = features_autoreg[idx - window_len : idx]
            window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(
                device, dtype=torch.float32
            )  # [1, window_len, 6]
            
            preds_seq = model(window_tensor)  # [1, window_len, 1]
            pred_delta = preds_seq[0, -1, 0].cpu().numpy()  # Last timestep prediction
            
            if predict_delta:
                preds_norm[idx] = preds_norm[idx - 1] + pred_delta
            else:
                preds_norm[idx] = pred_delta
            
            # Update features for next step (autoregressive: use prediction)
            features_autoreg[idx, 2] = preds_norm[idx]

    return preds_norm


def _gru_autoregressive_rollout_train_batch_tf(
    model: BatteryGRUWrapperTF,
    features_batch: torch.Tensor,  # [batch_size, max_length, 6]
    y_norm_batch: torch.Tensor,  # [batch_size, max_length]
    valid_mask: torch.Tensor,  # [batch_size, max_length]
    window_len: int,
    predict_delta: bool,
) -> torch.Tensor:
    """
    Teacher forcing: window_len만큼의 정답값을 사용해서 다음 타임스텝 하나만 예측.
    ResNet처럼 각 타임스텝마다 독립적으로 처리하되, window를 사용.
    
    예: window_len=20이면, k=0~19의 정답값을 보고, delta_Vcorr_pred(20)을 계산
    Vcorr_pred(21) = Vcorr(20) + delta_Vcorr_pred(20)
    
    Returns predictions: [batch_size, max_length]
    """
    batch_size, max_length, _ = features_batch.shape
    device = features_batch.device
    
    # Initialize predictions with initial values
    pred_y_norm_batch = torch.zeros_like(y_norm_batch)  # [batch_size, max_length]
    pred_y_norm_batch[:, :window_len] = y_norm_batch[:, :window_len]  # Copy initial window_len values
    
    # Find max valid length across all profiles
    valid_lengths = valid_mask.sum(dim=1)  # [batch_size]
    max_valid_length = valid_lengths.max().item()
    
    if max_valid_length <= window_len:
        return pred_y_norm_batch
    
    # Teacher forcing: 각 타임스텝마다 window_len만큼의 정답값을 사용해서 다음 타임스텝 하나만 예측
    # ResNet처럼 각 타임스텝마다 독립적으로 처리 (GPU 효율적)
    # 모든 프로파일을 배치로 처리하여 GPU 최대 활용
    for k in range(window_len, max_valid_length):
        # Build batch of windows: all profiles at timestep k
        # Window: [k-window_len : k] (정답값 사용)
        # clone()은 필요: Y 값을 수정해야 하므로
        windows_batch = features_batch[:, k - window_len : k, :].clone()  # [batch_size, window_len, 6]
        
        # Teacher forcing: use ground truth Y values in windows
        windows_batch[:, :, 2] = y_norm_batch[:, k - window_len : k]
        
        # Check which profiles are valid at this timestep
        valid_at_k = (k < valid_lengths) & ((k - window_len) >= 0)
        if not valid_at_k.any():
            continue
        
        # Predict delta for next timestep (k+1) for all profiles simultaneously
        preds_seq = model(windows_batch)  # [batch_size, window_len, 1]
        pred_delta_batch = preds_seq[:, -1, :].squeeze(-1)  # [batch_size] - 마지막 timestep의 예측값
        
        # Update predictions: Vcorr_pred(k+1) = Vcorr(k) + delta_Vcorr_pred(k)
        if predict_delta:
            pred_y_norm_batch[:, k] = torch.where(
                valid_at_k,
                y_norm_batch[:, k - 1] + pred_delta_batch,  # Teacher forcing: Vcorr(k)는 정답값 사용
                pred_y_norm_batch[:, k]
            )
        else:
            pred_y_norm_batch[:, k] = torch.where(
                valid_at_k,
                pred_delta_batch,
                pred_y_norm_batch[:, k]
            )
        
        # 메모리 정리
        del preds_seq
    
    return pred_y_norm_batch


def _batch_teacher_forcing_train_pass(
    model: BatteryGRUWrapperTF,
    profile_list: Sequence[Dict[str, Any]],
    window_len: int,
    predict_delta: bool,
    device: torch.device,
    training_batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Process multiple profiles in batch with teacher forcing.
    Teacher forcing: window_len만큼의 정답값을 사용해서 다음 타임스텝 하나만 예측.
    모든 프로파일을 배치로 처리하여 GPU 효율적으로 계산.
    
    1. 프로파일별로 GPU에 올림 (이게 batch)
    2. 각 프로파일 내에서 window를 슬라이딩하며 각 타임스텝마다 하나씩 예측
    3. 끝나면 Vpred 값들이 리턴됨
    4. 밖에서 loss 계산
    
    OOM 방지를 위해 프로파일을 작은 sub-batch로 나눠서 처리합니다.
    """
    # Prepare all profiles
    profile_data = []
    for profile in profile_list:
        feature_tensor, y_norm, total_steps, y_mean, y_std = _prepare_profile_tensors_tf(profile, device)
        if total_steps <= window_len:
            continue
        profile_data.append({
            "features": feature_tensor,  # [T, 6]
            "y_norm": y_norm,  # [T]
            "total_steps": total_steps,
            "y_mean": y_mean,
            "y_std": y_std,
        })
    
    if not profile_data:
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy_loss, {"num_windows": 0, "rmse_mV": float("nan")}
    
    # OOM 방지: 프로파일을 작은 sub-batch로 나눠서 처리
    # ResNet처럼 각 타임스텝마다 독립적으로 처리하므로 메모리 효율적
    # training_batch_size가 제공되면 그것을 sub_batch_size로 사용
    # ResNet처럼 프로파일 샘플링은 하지 않고, sub_batch_size로만 제어
    if training_batch_size is not None:
        # 사용자가 지정한 batch_size를 sub_batch_size로 사용
        sub_batch_size = min(training_batch_size, len(profile_data))
    else:
        # 기본값: ResNet과 비슷하게 큰 배치 사용 가능 (각 타임스텝마다 독립 처리)
        sub_batch_size = min(128, len(profile_data))
    
    all_losses: List[torch.Tensor] = []
    all_num_valid = 0
    all_sum_sq_norm = 0.0
    all_y_stds: List[float] = []
    
    # Process profiles in sub-batches
    for sub_batch_start in range(0, len(profile_data), sub_batch_size):
        sub_batch_end = min(sub_batch_start + sub_batch_size, len(profile_data))
        sub_batch_profiles = profile_data[sub_batch_start:sub_batch_end]
        
        # Prepare batch tensors with padding (like ResNet)
        batch_size = len(sub_batch_profiles)
        max_length = max(prof["total_steps"] for prof in sub_batch_profiles)
        
        # 메모리 최적화: 필요한 만큼만 할당
        features_batch = torch.zeros(batch_size, max_length, 6, dtype=torch.float32, device=device)
        y_norm_batch = torch.zeros(batch_size, max_length, dtype=torch.float32, device=device)
        valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        
        for i, prof in enumerate(sub_batch_profiles):
            valid_len = prof["total_steps"]
            features_batch[i, :valid_len] = prof["features"]
            y_norm_batch[i, :valid_len] = prof["y_norm"]
            valid_mask[i, :valid_len] = True
        
        # GPU 메모리 정리 (sub-batch 처리 전)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Perform batch autoregressive rollout with teacher forcing
        pred_y_norm_batch = _gru_autoregressive_rollout_train_batch_tf(
            model=model,
            features_batch=features_batch,
            y_norm_batch=y_norm_batch,
            valid_mask=valid_mask,
            window_len=window_len,
            predict_delta=predict_delta,
        )  # Shape: [batch_size, max_length]
        
        # Calculate loss on valid positions (exclude initial window_len)
        valid_target_mask = valid_mask[:, window_len:]  # [batch_size, max_length - window_len]
        pred_seq = pred_y_norm_batch[:, window_len:]  # [batch_size, max_length - window_len]
        target_seq = y_norm_batch[:, window_len:]  # [batch_size, max_length - window_len]
        
        # Calculate loss only on valid positions
        valid_target_mask_float = valid_target_mask.float()
        squared_diff = (pred_seq - target_seq) ** 2
        loss_V = (squared_diff * valid_target_mask_float).sum()
        num_valid = valid_target_mask.sum().item()
        
        if num_valid > 0:
            all_losses.append(loss_V)
            all_num_valid += num_valid
            all_sum_sq_norm += (squared_diff * valid_target_mask_float).sum().item()
            all_y_stds.extend([prof["y_std"] for prof in sub_batch_profiles])
        
        # Clear memory
        del features_batch, y_norm_batch, valid_mask, pred_y_norm_batch
    
    if not all_losses:
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy_loss, {"num_windows": 0, "rmse_mV": float("nan")}
    
    # Aggregate loss
    total_loss = sum(all_losses) / all_num_valid if all_num_valid > 0 else sum(all_losses)
    
    # Calculate RMSE
    avg_y_std = np.mean(all_y_stds) if all_y_stds else 1.0
    rmse_norm = math.sqrt(all_sum_sq_norm / all_num_valid) if all_num_valid > 0 else float("nan")
    rmse_mV = rmse_norm * avg_y_std * 1000
    
    stats = {
        "num_windows": all_num_valid,
        "rmse_mV": rmse_mV,
    }
    
    return total_loss, stats


def train_battery_gru_tf(
    profile_list: Sequence[Dict[str, Any]],
    num_epochs: int,
    lr: float,
    device: Union[str, torch.device],
    verbose: bool = True,
    pretrained_model_path: Optional[str] = None,
    window_len: int = 20,
    predict_delta: bool = True,
    shuffle_profiles: bool = True,
    early_stop_patience: int = 10,
    early_stop_window: int = 20,
    early_stop_delta_mV: float = 0.005,
    val_profile_list: Optional[Sequence[Dict[str, Any]]] = None,
    ar_val_interval: Optional[int] = None,
    hidden_size: int = 6,
    num_layers: int = 1,
    training_batch_size: Optional[int] = None,
) -> Tuple[BatteryGRUWrapperTF, Dict[str, Dict[str, Optional[float]]]]:
    """
    Train GRU with teacher forcing.
    Teacher forcing: window_len만큼의 정답값을 사용해서 다음 타임스텝 하나만 예측.
    모든 프로파일을 배치로 처리하여 GPU 효율적으로 계산.
    """
    
    device = torch.device(device)
    
    # GPU 메모리 정리 (트레이닝 시작 전)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if verbose:
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[GPU Memory] Cleared. Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    model = BatteryGRUWrapperTF(input_size=6, hidden_size=hidden_size, num_layers=num_layers, device=device).to(device)
    
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
        print("\n" + "=" * 70)
        print("GRU Teacher Forcing Training Configuration")
        print("=" * 70)
        print(f"Architecture : GRU (input=6, hidden={hidden_size}, layers={num_layers})")
        print(f"Head         : Linear({hidden_size} -> 1)")
        print(f"Device       : {device}")
        print(f"Epochs       : {num_epochs}")
        print(f"Learning rate: {lr}")
        print(f"Window len   : {window_len}")
        print(f"Predict delta: {predict_delta}")
        print(f"Train profiles: {len(profile_list)}")
        print(f"Val profiles  : {len(val_profile_list) if val_profile_list else 0}")
        print(f"Early-stop    : patience={early_stop_patience}, window={early_stop_window}, Δ={early_stop_delta_mV} mV")
        print("=" * 70 + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-3, threshold_mode="rel"
    )
    criterion = nn.MSELoss()

    history: Dict[str, Dict[str, Optional[float]]] = {"best": {}}
    best_rmse = float("inf")
    best_rmse_mV = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    rmse_window = deque(maxlen=early_stop_window)
    use_validation = val_profile_list is not None and len(val_profile_list) > 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        # GPU 메모리 정리 (각 epoch 시작 전)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if shuffle_profiles:
            profile_list_shuffled = list(profile_list)
            random.shuffle(profile_list_shuffled)
        else:
            profile_list_shuffled = list(profile_list)

        # Process all profiles in batch (GPU efficient)
        optimizer.zero_grad()
        loss_tensor, stats = _batch_teacher_forcing_train_pass(
            model=model,
            profile_list=profile_list_shuffled,
            window_len=window_len,
            predict_delta=predict_delta,
            device=device,
            training_batch_size=training_batch_size,  # Use provided batch_size
        )
        
        if stats["num_windows"] == 0:
            if verbose:
                print("No valid training windows for this epoch.")
            break
        
        loss_tensor.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()

        epoch_loss = loss_tensor.item()
        train_rmse_mV = stats["rmse_mV"]

        val_loss = None
        val_rmse_mV = None
        val_rmse_norm = None

        if use_validation:
            val_metrics = _evaluate_gru_profiles_tf(
                model=model,
                profile_list=val_profile_list or [],
                window_len=window_len,
                predict_delta=predict_delta,
                device=device,
            )
            val_rmse_mV = val_metrics["avg_rmse_vcorr_mV"]
            val_rmse_norm = val_metrics["avg_rmse_norm"]
            val_loss = val_rmse_norm**2 if val_rmse_norm == val_rmse_norm else None

        monitor_loss = val_loss if use_validation and val_loss is not None else epoch_loss
        scheduler.step(monitor_loss)

        monitor_rmse = (
            val_rmse_norm if use_validation and val_rmse_norm is not None else math.sqrt(epoch_loss)
        )
        monitor_rmse_mV = (
            val_rmse_mV if use_validation and val_rmse_mV is not None else train_rmse_mV
        )

        # Update best model if improved (same as ResNet: any improvement is accepted)
        # Early stopping uses rmse_window, not this condition
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

        epoch_elapsed_time = time.time() - epoch_start_time

        if verbose:
            msg = (
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Time: {epoch_elapsed_time:.1f}s | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Loss: {epoch_loss:.4e} | Train RMSE: {train_rmse_mV:.2f} mV"
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
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "early_stop_patience": early_stop_patience,
            "early_stop_window": early_stop_window,
            "early_stop_delta_mV": early_stop_delta_mV,
            "use_validation": use_validation,
        },
        "network_architecture": str(model),
    }
    best_model_path = f"best_model_gru_tf_rmse{best_rmse_mV:.2f}mV.pth"
    torch.save(checkpoint, best_model_path)
    if verbose:
        print(f"Best model saved to: {best_model_path}")

    history["best"]["checkpoint_path"] = best_model_path
    history["checkpoint_path"] = best_model_path

    return model, history


def _evaluate_gru_profiles_tf(
    model: BatteryGRUWrapperTF,
    profile_list: Sequence[Dict[str, Any]],
    window_len: int,
    predict_delta: bool,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate profiles with teacher forcing (batch processing, same as training).
    All profiles are processed simultaneously on GPU.
    """
    if not profile_list:
        return {
            "avg_rmse_vcorr_mV": float("nan"),
            "avg_rmse_norm": float("nan"),
            "avg_Y_std": float("nan"),
        }

    # Prepare all profiles
    profile_data = []
    for profile in profile_list:
        feature_tensor, y_norm, total_steps, y_mean, y_std = _prepare_profile_tensors_tf(profile, device)
        if total_steps <= window_len:
            continue
        profile_data.append({
            "features": feature_tensor,  # [T, 6]
            "y_norm": y_norm,  # [T]
            "total_steps": total_steps,
            "y_mean": y_mean,
            "y_std": y_std,
        })
    
    if not profile_data:
        return {
            "avg_rmse_vcorr_mV": float("nan"),
            "avg_rmse_norm": float("nan"),
            "avg_Y_std": float("nan"),
        }
    
    # OOM 방지: validation도 sub-batch로 나눠서 처리 (training과 동일)
    # ResNet처럼 각 타임스텝마다 독립적으로 처리하므로 큰 배치 사용 가능
    sub_batch_size = min(128, len(profile_data))
    
    all_rmse_norm: List[float] = []
    all_rmse_vcorr: List[float] = []
    all_y_stds: List[float] = []
    
    model.eval()
    with torch.no_grad():
        # Process profiles in sub-batches
        for sub_batch_start in range(0, len(profile_data), sub_batch_size):
            sub_batch_end = min(sub_batch_start + sub_batch_size, len(profile_data))
            sub_batch_profiles = profile_data[sub_batch_start:sub_batch_end]
            
            # Prepare batch tensors with padding
            batch_size = len(sub_batch_profiles)
            max_length = max(prof["total_steps"] for prof in sub_batch_profiles)
            
            features_batch = torch.zeros(batch_size, max_length, 6, dtype=torch.float32, device=device)
            y_norm_batch = torch.zeros(batch_size, max_length, dtype=torch.float32, device=device)
            valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
            
            for i, prof in enumerate(sub_batch_profiles):
                valid_len = prof["total_steps"]
                features_batch[i, :valid_len] = prof["features"]
                y_norm_batch[i, :valid_len] = prof["y_norm"]
                valid_mask[i, :valid_len] = True
            
            # Perform batch autoregressive rollout with teacher forcing (same as training)
            pred_y_norm_batch = _gru_autoregressive_rollout_train_batch_tf(
                model=model,
                features_batch=features_batch,
                y_norm_batch=y_norm_batch,
                valid_mask=valid_mask,
                window_len=window_len,
                predict_delta=predict_delta,
            )  # Shape: [batch_size, max_length]
            
            # Calculate loss on valid positions (exclude initial window_len)
            valid_target_mask = valid_mask[:, window_len:]  # [batch_size, max_length - window_len]
            pred_seq = pred_y_norm_batch[:, window_len:]  # [batch_size, max_length - window_len]
            target_seq = y_norm_batch[:, window_len:]  # [batch_size, max_length - window_len]
            
            # Calculate RMSE for each profile
            for i, prof in enumerate(sub_batch_profiles):
                valid_len = prof["total_steps"]
                valid_target_len = valid_len - window_len
                if valid_target_len <= 0:
                    continue
                
                pred_valid = pred_seq[i, :valid_target_len].cpu().numpy()
                target_valid = target_seq[i, :valid_target_len].cpu().numpy()
                
                rmse_norm = float(np.sqrt(np.mean((pred_valid - target_valid) ** 2)))
                rmse_vcorr = rmse_norm * prof["y_std"] * 1000
                
                all_rmse_norm.append(rmse_norm)
                all_rmse_vcorr.append(rmse_vcorr)
                all_y_stds.append(prof["y_std"])
    
    avg_rmse_norm = float(np.nanmean(all_rmse_norm)) if all_rmse_norm else float("nan")
    avg_rmse_vcorr = float(np.nanmean(all_rmse_vcorr)) if all_rmse_vcorr else float("nan")
    avg_y_std = float(np.nanmean(all_y_stds)) if all_y_stds else float("nan")
    
    return {
        "avg_rmse_vcorr_mV": avg_rmse_vcorr,
        "avg_rmse_norm": avg_rmse_norm,
        "avg_Y_std": avg_y_std,
    }


def train_gru_benchmark_tf(
    train_dict_list: Sequence[Dict[str, Any]],
    val_dict_list: Optional[Sequence[Dict[str, Any]]] = None,
    num_epochs: int = 1000,
    lr: float = 5e-4,
    device: Union[str, torch.device] = "cpu",
    pretrained_model_path: Optional[Union[str, Path]] = None,
    window_len: int = 20,
    batch_size: Optional[int] = None,
    predict_delta: bool = True,
    ar_val_interval: Optional[int] = None,
    verbose: bool = True,
    hidden_size: int = 6,
    num_layers: int = 1,
) -> Dict[str, Any]:
    """
    batch_size: Number of profiles to use per training step (None = use all, default 128)
    """
    model, history = train_battery_gru_tf(
        profile_list=train_dict_list,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        verbose=verbose,
        pretrained_model_path=pretrained_model_path,
        window_len=window_len,
        predict_delta=predict_delta,
        shuffle_profiles=True,
        early_stop_patience=10,
        early_stop_window=20,
        early_stop_delta_mV=0.005,
        val_profile_list=val_dict_list,
        ar_val_interval=ar_val_interval,
        hidden_size=hidden_size,
        num_layers=num_layers,
        training_batch_size=batch_size,  # Pass batch_size to control sub_batch_size
    )
    return {"model": model, "history": history}


def run_gru_benchmark_inference_tf(
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
    Run inference using BatteryGRUWrapperTF model (teacher forcing trained).
    Auto-regressive rollout over a list of profiles.
    """
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_info = checkpoint.get("training_info", {})

    inferred_window = training_info.get("window_len", window_len)
    if inferred_window is None:
        raise ValueError("window_len must be provided when checkpoint lacks metadata.")
    inferred_predict_delta = training_info.get("predict_delta", predict_delta)
    if inferred_predict_delta is None:
        inferred_predict_delta = True
    
    # Get model architecture from checkpoint
    inferred_hidden_size = training_info.get("hidden_size", hidden_size)
    if inferred_hidden_size is None:
        inferred_hidden_size = 32  # Default
    inferred_num_layers = training_info.get("num_layers", num_layers)
    if inferred_num_layers is None:
        inferred_num_layers = 1  # Default

    model = BatteryGRUWrapperTF(
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
        print(f"\n[GRU TF TEST] Loading checkpoint: {checkpoint_path}")
        if training_info:
            print("  --- Checkpoint Metadata ---")
            for key in sorted(training_info.keys()):
                print(f"  {key}: {training_info[key]}")
        else:
            print("  (no training_info metadata found)")
        print(f"[GRU TF TEST] Model: hidden_size={inferred_hidden_size}, num_layers={inferred_num_layers}")
        print(f"[GRU TF TEST] Total profiles to evaluate: {total_profiles}")
        print("=" * 60)

    for idx, profile in enumerate(dict_list):
        feature_tensor, y_norm, total_steps, y_mean, y_std = _prepare_profile_tensors_tf(profile, device)
        if total_steps <= inferred_window:
            if verbose:
                print(
                    f"[GRU TF TEST] Skipping profile {idx} "
                    f"(length {total_steps} <= window {inferred_window})"
                )
            continue

        # Autoregressive rollout (inference: use predictions, not ground truth)
        preds_norm = _gru_autoregressive_rollout_tf(
            model=model,
            feature_array=feature_tensor.cpu().numpy(),
            initial_y=y_norm.cpu().numpy(),
            window_len=inferred_window,
            predict_delta=inferred_predict_delta,
            device=device,
        )

        pred_vcorr = preds_norm * y_std + y_mean
        target_vcorr = y_norm.cpu().numpy() * y_std + y_mean
        
        # Get other profile data
        prepared = prepare_profile_for_gru(profile)
        v_spme = prepared["V_spme"]
        v_meas = prepared.get("V_meas")
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
                    np.sqrt(np.mean((vtotal_pred[mask_vtotal] - vtotal_target[mask_vtotal]) ** 2)) * 1000
                )
            else:
                rmse_vtotal = float("nan")
        else:
            rmse_vtotal = float("nan")

        results.append(
            {
                "profile_idx": idx,
                "time": time_array,
                "SOC": soc_array,
                "Vcorr_pred": pred_vcorr,
                "Vcorr_target": target_vcorr,
                "Vtotal_pred": vtotal_pred if v_meas is not None else None,
                "Vtotal_target": v_meas,
                "V_meas": v_meas,  # For plotting compatibility
                "V_spme": v_spme,
                "rmse_vcorr_mV": rmse_vcorr,
                "rmse_vtotal_mV": rmse_vtotal,
            }
        )
        rmse_vcorr_list.append(rmse_vcorr)
        rmse_vtotal_list.append(rmse_vtotal)
        y_std_list.append(y_std)

        if verbose and (idx + 1) % 50 == 0:
            print(f"[GRU TF TEST] Processed {idx + 1}/{total_profiles} profiles...")

    avg_rmse_vcorr = float(np.nanmean(rmse_vcorr_list)) if rmse_vcorr_list else float("nan")
    avg_rmse_vtotal = float(np.nanmean(rmse_vtotal_list)) if rmse_vtotal_list else float("nan")
    avg_y_std = float(np.nanmean(y_std_list)) if y_std_list else float("nan")

    metrics = {
        "avg_rmse_vcorr_mV": avg_rmse_vcorr,
        "avg_rmse_vtotal_mV": avg_rmse_vtotal,
        "avg_Y_std": avg_y_std,
        "num_profiles": len(results),
    }

    if verbose:
        print("=" * 60)
        print(f"[GRU TF TEST] Average RMSE (Vcorr): {avg_rmse_vcorr:.2f} mV")
        if not np.isnan(avg_rmse_vtotal):
            print(f"[GRU TF TEST] Average RMSE (Vtotal): {avg_rmse_vtotal:.2f} mV")
        print(f"[GRU TF TEST] Evaluated {len(results)} profiles")
        print("=" * 60)

    return results, metrics
