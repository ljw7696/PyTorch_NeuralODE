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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from GRU.battery_GRU_wrapper import prepare_profile_for_gru


class BatteryGRUWrapperTF(nn.Module):
    """
    GRU model wrapper for teacher forcing training.
    
    Architecture: Bidirectional GRU → Unidirectional GRU → Dense Layers → Linear(1)
    
    Input: [V_spme_norm, ocv, SOC, I, T, Y] (6 features) at timestep k
    Output: delta_Vcorr prediction (normalized) for timestep k+1
    
    Parameters:
        gru1_hidden: Hidden size for bidirectional GRU (0 to skip)
        gru2_hidden: Hidden size for unidirectional GRU (0 to skip)
        dense1_hidden: Hidden size for first dense layer (0 to skip)
        dense2_hidden: Hidden size for second dense layer (0 to skip)
    """

    def __init__(
        self,
        input_size: int = 6,
        gru1_hidden: int = 0,
        gru2_hidden: int = 96,
        dense1_hidden: int = 32,
        dense2_hidden: int = 12,
        dropout: float = 0.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        # Store architecture config for checkpoint
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
            # Determine input size for gru2
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
        
        # Dense layers
        self.use_dense1 = dense1_hidden > 0
        if self.use_dense1:
            self.dense1 = nn.Linear(dense_input_size, dense1_hidden)
        else:
            self.dense1 = None
        
        self.use_dense2 = dense2_hidden > 0
        if self.use_dense2:
            if self.use_dense1:
                dense2_input_size = dense1_hidden
            else:
                dense2_input_size = dense_input_size
            self.dense2 = nn.Linear(dense2_input_size, dense2_hidden)
        else:
            self.dense2 = None
        
        # Output layer (always present)
        if self.use_dense2:
            output_input_size = dense2_hidden
        elif self.use_dense1:
            output_input_size = dense1_hidden
        else:
            output_input_size = dense_input_size
        
        self.head = nn.Linear(output_input_size, 1)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0.0)

    def forward(self, inputs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            inputs: [batch_size, seq_len, 6] or [batch_size, 1, 6]
            hidden: Optional tuple of (gru1_hidden, gru2_hidden) for state continuation
        
        Returns:
            outputs: [batch_size, seq_len, 1] - delta_Vcorr predictions
            hidden: Tuple of (gru1_hidden, gru2_hidden) for next timestep
        """
        batch_size = inputs.shape[0]
        
        # Handle empty batch
        if batch_size == 0:
            # Return empty outputs and dummy hidden states
            seq_len = inputs.shape[1]
            dummy_output = torch.zeros(0, seq_len, 1, device=inputs.device, dtype=inputs.dtype)
            dummy_gru1_hidden = torch.zeros(2, 0, 1, device=inputs.device, dtype=inputs.dtype)
            dummy_gru2_hidden = torch.zeros(1, 0, 1, device=inputs.device, dtype=inputs.dtype)
            return dummy_output, (dummy_gru1_hidden, dummy_gru2_hidden)
        
        x = inputs
        
        gru1_hidden = None
        gru2_hidden = None
        
        # Bidirectional GRU
        if self.use_gru1:
            if hidden is not None:
                gru1_hidden_prev, _ = hidden
                x, gru1_hidden = self.gru1(x, gru1_hidden_prev)
            else:
                x, gru1_hidden = self.gru1(x)  # [batch_size, seq_len, gru1_hidden * 2]
        
        # Unidirectional GRU
        if self.use_gru2:
            if hidden is not None:
                _, gru2_hidden_prev = hidden
                x, gru2_hidden = self.gru2(x, gru2_hidden_prev)
            else:
                x, gru2_hidden = self.gru2(x)  # [batch_size, seq_len, gru2_hidden]
        
        # If no GRU layers, create dummy hidden states
        if gru1_hidden is None:
            if self.use_gru1:
                num_layers1 = self.gru1.num_layers
                hidden_size1 = self.gru1.hidden_size
                gru1_hidden = torch.zeros(
                    num_layers1 * 2, batch_size, hidden_size1,
                    device=x.device, dtype=x.dtype
                )
            else:
                gru1_hidden = torch.zeros(2, batch_size, 1, device=x.device, dtype=x.dtype)
        
        if gru2_hidden is None:
            if self.use_gru2:
                num_layers2 = self.gru2.num_layers
                hidden_size2 = self.gru2.hidden_size
                gru2_hidden = torch.zeros(
                    num_layers2, batch_size, hidden_size2,
                    device=x.device, dtype=x.dtype
                )
            else:
                gru2_hidden = torch.zeros(1, batch_size, 1, device=x.device, dtype=x.dtype)
        
        # Apply dense layers to each timestep
        feature_dim = x.shape[-1]
        x_flat = x.reshape(-1, feature_dim)
        
        # Dense layers
        if self.use_dense1:
            x_flat = self.activation(self.dense1(x_flat))
            if self.dropout is not None:
                x_flat = self.dropout(x_flat)
        
        if self.use_dense2:
            x_flat = self.activation(self.dense2(x_flat))
            if self.dropout is not None:
                x_flat = self.dropout(x_flat)
        
        # Output layer
        preds_flat = self.head(x_flat)  # [batch * seq_len, 1]
        
        # Reshape back to [batch, seq_len, 1]
        seq_len = inputs.shape[1]
        outputs = preds_flat.reshape(batch_size, seq_len, 1)
        
        return outputs, (gru1_hidden, gru2_hidden)


def _prepare_profile_tensors_tf(profile: Dict[str, Any], device: torch.device) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """Prepare profile data for teacher forcing training."""
    prepared = prepare_profile_for_gru(profile)

    # Build feature array: [V_spme_norm, ocv, SOC, I, T, Y] at timestep k
    feature_array = np.stack(
        [
            prepared["V_spme_norm"],
            prepared["ocv"],
            prepared["SOC"],
            prepared["I"],
            prepared["T"],
            prepared["Y"],  # Teacher forcing: use true Y(k) value
        ],
        axis=-1,
    )

    feature_tensor = torch.from_numpy(feature_array).to(device, dtype=torch.float32)  # [T, 6]
    y_norm = torch.from_numpy(prepared["Y"]).to(device, dtype=torch.float32)  # [T]
    
    return prepared, feature_tensor, y_norm


def _calculate_autoregressive_rmse_tf(
    model: BatteryGRUWrapperTF,
    profile_list: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Calculate auto-regressive RMSE for validation profiles.
    Each profile is processed independently with full auto-regressive rollout.
    """
    model.eval()
    
    if not profile_list:
        return {"avg_rmse_mV": float("nan"), "rmse_list": []}
    
    rmse_list: List[float] = []
    y_std_list: List[float] = []
    pred_all: List[np.ndarray] = []
    target_all: List[np.ndarray] = []
    time_all: List[np.ndarray] = []
    
    with torch.no_grad():
        for profile in profile_list:
            prepared, feature_tensor, y_norm = _prepare_profile_tensors_tf(profile, device)
            total_steps = y_norm.shape[0]
            
            if total_steps <= 1:
                continue
            
            # Initialize predictions
            pred_y_norm = y_norm.clone()  # Start with true values
            features_autoreg = feature_tensor.clone()  # [T, 6]
            
            # Initialize hidden states
            gru1_hidden = None
            gru2_hidden = None
            
            # Auto-regressive rollout: k=0 to k=total_steps-2 (predict k+1)
            for k in range(total_steps - 1):
                # Get input at timestep k: [1, 1, 6]
                input_k = features_autoreg[k:k+1, :].unsqueeze(0)  # [1, 1, 6]
                # Use predicted value (auto-regressive)
                input_k[0, 0, 5] = pred_y_norm[k]  # Y(k) = predicted value
                
                # Forward pass
                if gru1_hidden is None:
                    preds_seq, (gru1_hidden, gru2_hidden) = model(input_k, hidden=None)
                else:
                    preds_seq, (gru1_hidden, gru2_hidden) = model(input_k, hidden=(gru1_hidden, gru2_hidden))
                
                # Get prediction: delta_Vcorr(k+1)
                pred_delta = preds_seq[0, 0, 0].item()
                
                # Update predicted Y(k+1)
                pred_y_norm[k + 1] = pred_y_norm[k] + pred_delta
                features_autoreg[k + 1, 5] = pred_y_norm[k + 1]
            
            # Calculate RMSE (exclude first timestep)
            if total_steps > 1:
                pred_valid = pred_y_norm[1:].cpu().numpy()
                target_valid = y_norm[1:].cpu().numpy()
                time_valid = prepared["time"][1:]  # Time array (exclude first timestep)
                
                rmse_norm = float(np.sqrt(np.mean((pred_valid - target_valid) ** 2)))
                y_std = prepared["Y_std"]
                rmse_mV = rmse_norm * y_std * 1000
                
                rmse_list.append(rmse_mV)
                y_std_list.append(y_std)
                
                # Store for time-segmented RMSE calculation
                pred_all.append(pred_valid)
                target_all.append(target_valid)
                time_all.append(time_valid)
    
    avg_rmse_mV = float(np.nanmean(rmse_list)) if rmse_list else float("nan")
    
    # Calculate time-segmented RMSE
    time_segments = [30, 60, 120, 180, 300, 600, 900, 1200, float("inf")]
    time_segmented_rmse = {}
    
    if pred_all:
        pred_flat = np.concatenate(pred_all)
        target_flat = np.concatenate(target_all)
        time_flat = np.concatenate(time_all)
        
        for t_max in time_segments:
            mask = time_flat <= t_max
            if mask.any():
                rmse_seg = float(np.sqrt(np.mean((pred_flat[mask] - target_flat[mask]) ** 2)))
                avg_y_std = float(np.nanmean(y_std_list)) if y_std_list else 1.0
                rmse_seg_mV = rmse_seg * avg_y_std * 1000
                label = f"{int(t_max)}s" if t_max != float("inf") else "all"
                time_segmented_rmse[label] = rmse_seg_mV
        
        # Calculate rest RMSE (after 1730s)
        mask_rest = time_flat > 1730
        if mask_rest.any():
            rmse_rest = float(np.sqrt(np.mean((pred_flat[mask_rest] - target_flat[mask_rest]) ** 2)))
            avg_y_std = float(np.nanmean(y_std_list)) if y_std_list else 1.0
            rmse_rest_mV = rmse_rest * avg_y_std * 1000
            time_segmented_rmse["rest_after_1730s"] = rmse_rest_mV
    
    return {
        "avg_rmse_mV": avg_rmse_mV,
        "rmse_list": rmse_list,
        "time_segmented_rmse": time_segmented_rmse,
    }


def _train_pass_tbptt_tf(
    model: BatteryGRUWrapperTF,
    profile_list: Sequence[Dict[str, Any]],
    device: torch.device,
    tbptt_length: int = 200,
    verbose: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    teacher_forcing_ratio: float = 1.0,
) -> Dict[str, float]:
    """
    TBPTT training pass for teacher forcing with scheduled sampling.
    
    Input: x(k) = [V_spme_norm(k), ocv(k), SOC(k), I(k), T(k), Y(k)] at timestep k
    Target: Y(k+1) (next timestep)
    Hidden state flows from t=0 to profile end.
    
    Args:
        teacher_forcing_ratio: Probability of using true value (1.0 = pure TF, 0.0 = pure autoregressive)
    """
    model.train()
    
    if not profile_list:
        return {"num_steps": 0, "rmse_mV": float("nan"), "total_loss": 0.0}
    
    # Prepare all profiles
    profile_data = []
    for profile in profile_list:
        prepared, feature_tensor, y_norm = _prepare_profile_tensors_tf(profile, device)
        total_steps = y_norm.shape[0]
        if total_steps <= 1:
            continue  # Need at least 2 timesteps (k and k+1)
        profile_data.append({
            "prepared": prepared,
            "features": feature_tensor,  # [total_steps, 6]
            "y_norm": y_norm,  # [total_steps]
            "total_steps": total_steps,
        })
    
    if not profile_data:
        return {"num_steps": 0, "rmse_mV": float("nan"), "total_loss": 0.0}
    
    # Pad profiles to same length for batch processing
    max_length = max(p["total_steps"] for p in profile_data)
    batch_size_actual = len(profile_data)
    
    # Create batch tensors
    features_batch = torch.zeros(batch_size_actual, max_length, 6, device=device)
    y_norm_batch = torch.zeros(batch_size_actual, max_length, device=device)
    valid_mask = torch.zeros(batch_size_actual, max_length, dtype=torch.bool, device=device)
    pred_y_norm_batch = torch.zeros(batch_size_actual, max_length, device=device)  # For scheduled sampling
    y_std_list = []
    
    for i, p in enumerate(profile_data):
        valid_len = p["total_steps"]
        features_batch[i, :valid_len] = p["features"]
        y_norm_batch[i, :valid_len] = p["y_norm"]
        pred_y_norm_batch[i, :valid_len] = p["y_norm"]  # Initialize with true values
        valid_mask[i, :valid_len] = True
        y_std_list.append(p["prepared"]["Y_std"])
    
    # Find max valid length
    valid_lengths = valid_mask.sum(dim=1)  # [batch_size]
    max_valid_length = valid_lengths.max().item()
    
    if max_valid_length <= 1:
        return {"num_steps": 0, "rmse_mV": float("nan"), "total_loss": 0.0}
    
    # Initialize hidden states (will be set on first forward pass)
    gru1_hidden = None
    gru2_hidden = None
    
    # TBPTT: Process in chunks
    chunk_losses = []
    total_valid_steps = 0
    
    # Process from k=0 to k=max_valid_length-2 (since target is k+1)
    # We can predict k+1 for k in range(0, max_valid_length-1)
    for chunk_start in range(0, max_valid_length - 1, tbptt_length):
        chunk_end = min(chunk_start + tbptt_length, max_valid_length - 1)
        chunk_losses_timestep = []
        
        # Detach hidden state at chunk start to break computation graph
        if gru1_hidden is not None:
            gru1_hidden = gru1_hidden.detach()
            gru2_hidden = gru2_hidden.detach()
        
        # Forward pass for this chunk
        for k in range(chunk_start, chunk_end):
            # Check which profiles are valid at this timestep and have next timestep
            valid_at_k = (k < valid_lengths) & ((k + 1) < valid_lengths)
            if not valid_at_k.any():
                continue
            
            # Get input at timestep k: [batch_size, 1, 6]
            # x(k) = [V_spme_norm(k), ocv(k), SOC(k), I(k), T(k), Y(k)]
            input_k = features_batch[:, k:k+1, :].clone()  # [batch_size, 1, 6]
            
            # Scheduled sampling: use true value or predicted value based on ratio
            if k > 0 and teacher_forcing_ratio < 1.0:
                use_true = torch.rand(batch_size_actual, device=device) < teacher_forcing_ratio
                input_k[:, 0, 5] = torch.where(
                    use_true,
                    y_norm_batch[:, k],  # True value
                    pred_y_norm_batch[:, k]  # Predicted value
                )
            else:
                # k=0 or pure TF: always use true value
                input_k[:, 0, 5] = y_norm_batch[:, k]
            
            # Forward pass with continuing hidden state
            if gru1_hidden is None:
                preds_seq, (gru1_hidden_new, gru2_hidden_new) = model(input_k, hidden=None)
                gru1_hidden = gru1_hidden_new
                gru2_hidden = gru2_hidden_new
            else:
                preds_seq, (gru1_hidden_new, gru2_hidden_new) = model(input_k, hidden=(gru1_hidden, gru2_hidden))
                # Update hidden state only for valid profiles
                valid_mask_hidden = valid_at_k.float().unsqueeze(0).unsqueeze(-1)
                gru1_hidden = gru1_hidden * (1 - valid_mask_hidden) + gru1_hidden_new * valid_mask_hidden
                gru2_hidden = gru2_hidden * (1 - valid_mask_hidden) + gru2_hidden_new * valid_mask_hidden
            
            # Get prediction: delta_Vcorr(k+1)
            pred_delta = preds_seq[:, 0, 0]  # [batch_size]
            
            # Update predicted Y(k+1) for next timestep (for scheduled sampling)
            pred_y_norm_batch[:, k + 1] = torch.where(
                valid_at_k,
                pred_y_norm_batch[:, k] + pred_delta.detach(),  # Y_pred(k+1) = Y_pred(k) + delta_pred
                pred_y_norm_batch[:, k + 1]
            )
            
            # Target: Y(k+1) - Y(k) = delta_Y(k+1)
            target_delta = y_norm_batch[:, k + 1] - y_norm_batch[:, k]  # [batch_size]
            
            # Calculate loss
            loss_per_profile = (pred_delta - target_delta) ** 2
            valid_loss = loss_per_profile * valid_at_k.float()
            timestep_loss = valid_loss.sum() / max(valid_at_k.sum().item(), 1)
            chunk_losses_timestep.append(timestep_loss)
            total_valid_steps += valid_at_k.sum().item()
        
        # Backward pass for entire chunk
        if chunk_losses_timestep:
            chunk_loss = torch.stack(chunk_losses_timestep).mean()
            
            # Only do backward pass if optimizer is provided (training mode)
            if optimizer is not None:
                optimizer.zero_grad()
                chunk_loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Detach hidden state after backward
                if gru1_hidden is not None:
                    gru1_hidden = gru1_hidden.detach()
                    gru2_hidden = gru2_hidden.detach()
            
            # Always append loss for RMSE calculation (both training and validation)
            chunk_losses.append(chunk_loss.item())
    
    # Calculate RMSE
    total_loss = sum(chunk_losses) if chunk_losses else 0.0
    
    # For RMSE calculation, we need to reconstruct predictions
    # This is approximate - for exact RMSE, we'd need to run inference
    if total_valid_steps > 0:
        avg_loss = total_loss / len(chunk_losses) if chunk_losses else 0.0
        rmse_norm = math.sqrt(avg_loss)
        avg_y_std = float(np.mean(y_std_list))
        rmse_mV = rmse_norm * avg_y_std * 1000
    else:
        rmse_mV = float("nan")
    
    return {
        "num_steps": total_valid_steps,
        "rmse_mV": rmse_mV,
        "total_loss": total_loss,
    }


def train_gru_tf(
    train_dict_list: Sequence[Dict[str, Any]],
    val_dict_list: Optional[Sequence[Dict[str, Any]]] = None,
    num_epochs: int = 1000,
    lr: float = 5e-4,
    device: Union[str, torch.device] = "cpu",
    early_stop_window: int = 20,
    verbose: bool = True,
    gru1_hidden: int = 0,
    gru2_hidden: int = 96,
    dense1_hidden: int = 32,
    dense2_hidden: int = 12,
    tbptt_length: int = 200,
    p_final: float = 1.0,
) -> Tuple[BatteryGRUWrapperTF, Dict[str, Any]]:
    """
    Train GRU with teacher forcing and TBPTT.
    
    Args:
        train_dict_list: List of training profiles (dicts)
        val_dict_list: Optional list of validation profiles
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to use
        early_stop_window: Window size for early stopping
        verbose: Print training progress
        gru1_hidden: Bidirectional GRU hidden size (0 to skip)
        gru2_hidden: Unidirectional GRU hidden size (0 to skip)
        dense1_hidden: First dense layer hidden size (0 to skip)
        dense2_hidden: Second dense layer hidden size (0 to skip)
        tbptt_length: TBPTT chunk length
        p_final: Final teacher forcing ratio for exponential scheduled sampling (1.0 = pure TF)
    
    Returns:
        Trained model and training history
    """
    device = torch.device(device)
    
    # Initialize model
    model = BatteryGRUWrapperTF(
        input_size=6,
        gru1_hidden=gru1_hidden,
        gru2_hidden=gru2_hidden,
        dense1_hidden=dense1_hidden,
        dense2_hidden=dense2_hidden,
        device=device,
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-3, threshold_mode="rel"
    )

    history: Dict[str, Any] = {
        "best": {},
        "epochs": [],
        "config": {
            "num_epochs": num_epochs,
            "lr": lr,
            "device": str(device),
            "early_stop_window": early_stop_window,
            "gru1_hidden": gru1_hidden,
            "gru2_hidden": gru2_hidden,
            "dense1_hidden": dense1_hidden,
            "dense2_hidden": dense2_hidden,
            "tbptt_length": tbptt_length,
            "p_final": p_final,
            "architecture_config": model.architecture_config,
        }
    }
    
    # Track best models for different criteria
    best_models: Dict[str, Dict[str, Any]] = {
        "60s": {"epoch": -1, "early_rmse": float("inf"), "rest_rmse": float("inf"), "state_dict": None},
        "120s": {"epoch": -1, "early_rmse": float("inf"), "rest_rmse": float("inf"), "state_dict": None},
        "300s": {"epoch": -1, "early_rmse": float("inf"), "rest_rmse": float("inf"), "state_dict": None},
        "1200s": {"epoch": -1, "early_rmse": float("inf"), "rest_rmse": float("inf"), "state_dict": None},
        "overall": {"epoch": -1, "rmse": float("inf"), "state_dict": None},
    }
    best_rmse = float("inf")
    best_rmse_mV = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    rmse_window = deque(maxlen=early_stop_window)
    use_validation = val_dict_list is not None and len(val_dict_list) > 0
    
    # Fixed subset of training data for evaluation when validation is not available
    train_eval_subset = None
    if not use_validation and len(train_dict_list) > 0:
        # Select 4 samples at 12.5, 37.5, 62.5, 87.5 percentiles
        n = len(train_dict_list)
        percentiles = [12.5, 37.5, 62.5, 87.5]
        indices = [int(n * p / 100.0) for p in percentiles]
        # Ensure indices are within bounds and unique
        indices = [min(idx, n - 1) for idx in indices]
        indices = sorted(list(set(indices)))  # Remove duplicates and sort
        train_eval_subset = [train_dict_list[i] for i in indices]

    if verbose:
        print("\n" + "=" * 70)
        print("GRU Teacher Forcing Training Configuration")
        print("=" * 70)
        print(f"Architecture: GRU1(bidir={gru1_hidden}) → GRU2({gru2_hidden}) → Dense1({dense1_hidden}) → Dense2({dense2_hidden})")
        print(f"Device       : {device}")
        print(f"Epochs       : {num_epochs}")
        print(f"Learning rate: {lr}")
        print(f"TBPTT length : {tbptt_length}")
        print(f"Train profiles: {len(train_dict_list)}")
        print(f"Val profiles  : {len(val_dict_list) if val_dict_list else 0}")
        print(f"Early-stop    : window={early_stop_window}")
        print("=" * 70 + "\n")

    # Calculate exponential scheduled sampling decay factor
    gamma = p_final ** (1.0 / num_epochs) if p_final > 0 else 0.0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Calculate teacher forcing ratio for this epoch (exponential decay)
        teacher_forcing_ratio = gamma ** epoch if p_final < 1.0 else 1.0
        
        if verbose:
            print(f"  Teacher forcing ratio: {teacher_forcing_ratio:.4f} ({teacher_forcing_ratio*100:.2f}% true, {((1-teacher_forcing_ratio)*100):.2f}% predicted)")
        
        # Shuffle training data
        train_list_shuffled = list(train_dict_list)
        random.shuffle(train_list_shuffled)
        
        # Training pass (optimizer.zero_grad() and optimizer.step() are called inside TBPTT chunks)
        train_stats = _train_pass_tbptt_tf(
            model=model,
            profile_list=train_list_shuffled,
            device=device,
            tbptt_length=tbptt_length,
            verbose=verbose and epoch == 0,
            optimizer=optimizer,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        
        train_rmse_mV = train_stats["rmse_mV"]
        train_loss = train_stats["total_loss"]
        
        # Validation
        val_rmse_mV = None
        val_loss = None
        val_ar_rmse_mV = None  # Auto-regressive RMSE
        if use_validation:
            model.eval()
            with torch.no_grad():
                # Teacher forcing RMSE (for comparison)
                val_stats = _train_pass_tbptt_tf(
                model=model,
                    profile_list=val_dict_list or [],
                device=device,
                    tbptt_length=tbptt_length,
                    verbose=False,
                    teacher_forcing_ratio=1.0,  # Validation: always use true values
                )
                val_rmse_mV = val_stats["rmse_mV"]
                val_loss = val_stats["total_loss"]
                
                # Auto-regressive RMSE (actual inference scenario)
                val_ar_stats = _calculate_autoregressive_rmse_tf(
                model=model,
                    profile_list=val_dict_list or [],
                device=device,
            )
                val_ar_rmse_mV = val_ar_stats["avg_rmse_mV"]
                val_ar_time_segmented = val_ar_stats.get("time_segmented_rmse", {})
                
                # Update best models based on criteria
                if val_ar_time_segmented:
                    rest_rmse = val_ar_time_segmented.get("rest_after_1730s", float("inf"))
                    overall_rmse = val_ar_time_segmented.get("all", float("inf"))
                    
                    # Check each early segment
                    for seg_name in ["60s", "120s", "300s", "1200s"]:
                        early_rmse = val_ar_time_segmented.get(seg_name, float("inf"))
                        if early_rmse == float("inf") or rest_rmse == float("inf"):
                            continue
                        
                        current_best = best_models[seg_name]
                        threshold = current_best["early_rmse"] * 1.2  # 20% threshold
                        
                        # First epoch or (within threshold and better rest RMSE)
                        if (current_best["epoch"] == -1) or (early_rmse <= threshold and rest_rmse < current_best["rest_rmse"]):
                            best_models[seg_name] = {
                                "epoch": epoch + 1,
                                "early_rmse": early_rmse,
                                "rest_rmse": rest_rmse,
                                "state_dict": copy.deepcopy(model.state_dict()),
                            }
                    
                    # Overall best
                    if overall_rmse < best_models["overall"]["rmse"]:
                        best_models["overall"] = {
                            "epoch": epoch + 1,
                            "rmse": overall_rmse,
                            "state_dict": copy.deepcopy(model.state_dict()),
                        }
        else:
            # No validation: use fixed subset of training data for autoregressive evaluation
            model.eval()
            with torch.no_grad():
                train_ar_stats = _calculate_autoregressive_rmse_tf(
                    model=model,
                    profile_list=train_eval_subset,
                    device=device,
                )
                val_ar_rmse_mV = train_ar_stats["avg_rmse_mV"]
                val_ar_time_segmented = train_ar_stats.get("time_segmented_rmse", {})
                
                # Update best models based on criteria (same logic as validation)
                if val_ar_time_segmented:
                    rest_rmse = val_ar_time_segmented.get("rest_after_1730s", float("inf"))
                    overall_rmse = val_ar_time_segmented.get("all", float("inf"))
                    
                    # Check each early segment
                    for seg_name in ["60s", "120s", "300s", "1200s"]:
                        early_rmse = val_ar_time_segmented.get(seg_name, float("inf"))
                        if early_rmse == float("inf") or rest_rmse == float("inf"):
                            continue
                        
                        current_best = best_models[seg_name]
                        threshold = current_best["early_rmse"] * 1.2  # 20% threshold
                        
                        # First epoch or (within threshold and better rest RMSE)
                        if (current_best["epoch"] == -1) or (early_rmse <= threshold and rest_rmse < current_best["rest_rmse"]):
                            best_models[seg_name] = {
                                "epoch": epoch + 1,
                                "early_rmse": early_rmse,
                                "rest_rmse": rest_rmse,
                                "state_dict": copy.deepcopy(model.state_dict()),
                            }
                    
                    # Overall best
                    if overall_rmse < best_models["overall"]["rmse"]:
                        best_models["overall"] = {
                            "epoch": epoch + 1,
                            "rmse": overall_rmse,
                            "state_dict": copy.deepcopy(model.state_dict()),
                        }

        # Learning rate scheduling
        monitor_loss = val_loss if use_validation and val_loss is not None else train_loss
        scheduler.step(monitor_loss)

        # Update best model (use auto-regressive RMSE if available, otherwise teacher forcing)
        if val_ar_rmse_mV is not None:
            monitor_rmse_mV = val_ar_rmse_mV  # Use auto-regressive RMSE
        elif use_validation and val_rmse_mV is not None:
            monitor_rmse_mV = val_rmse_mV  # Fallback to validation teacher forcing RMSE
        else:
            monitor_rmse_mV = train_rmse_mV  # Fallback to training teacher forcing RMSE
        
        if monitor_rmse_mV < best_rmse_mV and monitor_rmse_mV == monitor_rmse_mV:  # Check for NaN
            best_rmse_mV = monitor_rmse_mV
            best_rmse = math.sqrt(monitor_loss) if monitor_loss > 0 else float("inf")
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_since_improve = 0
            history["best"] = {
                "epoch": best_epoch,
                "train_loss": train_loss,
                "train_rmse_mV": train_rmse_mV,
                "val_loss": val_loss,
                "val_rmse_mV": val_rmse_mV,
                "val_ar_rmse_mV": val_ar_rmse_mV,  # Auto-regressive RMSE
            }
            if verbose:
                print(f"✅ Best RMSE: {monitor_rmse_mV:.2f} mV at epoch {epoch + 1}")
        else:
            epochs_since_improve += 1

        epoch_elapsed_time = time.time() - epoch_start_time

        # Store epoch information
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_rmse_mV": train_rmse_mV,
            "val_loss": val_loss,
            "val_rmse_mV": val_rmse_mV,
            "val_ar_rmse_mV": val_ar_rmse_mV,
            "val_ar_time_segmented": val_ar_time_segmented if val_ar_time_segmented else {},
            "lr": optimizer.param_groups[0]['lr'],
            "time_elapsed": epoch_elapsed_time,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }
        history["epochs"].append(epoch_info)

        if verbose:
            msg = (
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Time: {epoch_elapsed_time:.1f}s | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Loss: {train_loss:.4e} | Train RMSE: {train_rmse_mV:.2f} mV"
            )
            if use_validation and val_rmse_mV is not None:
                msg += f" | Val RMSE (TF): {val_rmse_mV:.2f} mV"
            if val_ar_rmse_mV is not None:
                label = "Val" if use_validation else "Train"
                msg += f" | {label} RMSE (AR): {val_ar_rmse_mV:.2f} mV"
            print(msg)

            # Print time-segmented autoregressive RMSE
            if val_ar_time_segmented:
                time_segments = ["30s", "60s", "120s", "180s", "300s", "600s", "900s", "1200s", "all"]
                seg_strs = [f"{seg}: {val_ar_time_segmented.get(seg, float('nan')):.2f}mV" for seg in time_segments if seg in val_ar_time_segmented]
                if seg_strs:
                    print(f"  Val RMSE (AR) by time: {' | '.join(seg_strs)}")

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            base_info = {
                "best_rmse_mV": best_rmse_mV,
                "best_rmse_norm": best_rmse,
                "best_epoch": best_epoch,
                "current_epoch": epoch + 1,
                "total_epochs": num_epochs,
                "lr": lr,
                "tbptt_length": tbptt_length,
                "architecture_config": model.architecture_config,
            }
            
            # Save current model with epoch number
            checkpoint_path = f"best_model_gru_tf_epoch{epoch + 1}_rmse{monitor_rmse_mV:.2f}mV.pth"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "training_info": {**base_info},
                "history": history,
            }
            torch.save(checkpoint, checkpoint_path)
            if verbose:
                print(f"Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if early_stop_window > 0 and train_rmse_mV == train_rmse_mV:
            rmse_window.append(train_rmse_mV)
            if (
                rmse_window.maxlen == len(rmse_window)
                and (max(rmse_window) - min(rmse_window)) <= 0.005
            ):
                if verbose:
                    print(f"Early stopping triggered (ΔRMSE <= 0.005 mV over {early_stop_window} epochs).")
                break

    # Save last epoch model before loading best model
    if len(history["epochs"]) > 0:
        last_epoch_info = history["epochs"][-1]
        last_epoch_num = last_epoch_info["epoch"]
        if last_epoch_num % 100 != 0:  # Only save if not already saved at 100 epoch interval
            last_epoch_rmse = last_epoch_info.get("val_ar_rmse_mV") or last_epoch_info.get("train_rmse_mV", 0.0)
            base_info_last = {
                "best_rmse_mV": best_rmse_mV,
                "best_rmse_norm": best_rmse,
                "best_epoch": best_epoch,
                "current_epoch": last_epoch_num,
                "total_epochs": num_epochs,
                "lr": lr,
                "tbptt_length": tbptt_length,
                "architecture_config": model.architecture_config,
            }
            last_epoch_path = f"best_model_gru_tf_epoch{last_epoch_num}_rmse{last_epoch_rmse:.2f}mV.pth"
            checkpoint_last = {
                "model_state_dict": model.state_dict(),
                "training_info": {**base_info_last},
                "history": history,
            }
            torch.save(checkpoint_last, last_epoch_path)
            if verbose:
                print(f"Last epoch model saved to: {last_epoch_path}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    if best_epoch == -1:
        best_epoch = history["best"].get("epoch", num_epochs) if history["best"] else num_epochs

    # Save checkpoints (final)
    base_info = {
            "best_rmse_mV": best_rmse_mV,
            "best_rmse_norm": best_rmse,
            "best_epoch": best_epoch,
            "current_epoch": num_epochs,
            "total_epochs": num_epochs,
            "lr": lr,
        "tbptt_length": tbptt_length,
        "architecture_config": model.architecture_config,
    }
    
    # Save best model (overall) with epoch number
    best_model_path = f"best_model_gru_tf_epoch{best_epoch}_rmse{best_rmse_mV:.2f}mV.pth"
    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "training_info": {**base_info},
        "history": history,
    }
    torch.save(checkpoint, best_model_path)
    if verbose:
        print(f"Best model saved to: {best_model_path}")
    
    # Save criteria-based best models
    saved_paths = [best_model_path]
    for seg_name, best_info in best_models.items():
        if best_info["epoch"] != -1 and best_info["state_dict"] is not None:
            if seg_name == "overall":
                continue  # Already saved above
            path = f"best_model_gru_tf_epoch{best_info['epoch']}_{seg_name}_early{best_info['early_rmse']:.2f}mV_rest{best_info['rest_rmse']:.2f}mV.pth"
            checkpoint = {
                "model_state_dict": best_info["state_dict"],
                "training_info": {
                    **base_info,
                    "criteria": seg_name,
                    "early_rmse_mV": best_info["early_rmse"],
                    "rest_rmse_mV": best_info["rest_rmse"],
                },
                "history": history,
            }
            torch.save(checkpoint, path)
            saved_paths.append(path)
            if verbose:
                print(f"  {seg_name} best (epoch {best_info['epoch']}): {path}")

    history["best"]["checkpoint_path"] = best_model_path
    history["checkpoint_path"] = best_model_path
    history["best"]["criteria_checkpoints"] = {k: f"best_model_gru_tf_{k}_..." for k in best_models.keys() if k != "overall" and best_models[k]["epoch"] != -1}

    return model, history


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[BatteryGRUWrapperTF, Dict[str, Any]]:
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        device: Device to load model on
    
    Returns:
        model: Loaded BatteryGRUWrapperTF model
        training_info: Training information from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    training_info = checkpoint.get("training_info", {})
    architecture_config = training_info.get("architecture_config", {})
    
    # Create model with saved architecture
    model = BatteryGRUWrapperTF(
        input_size=architecture_config.get("input_size", 6),
        gru1_hidden=architecture_config.get("gru1_hidden", 0),
        gru2_hidden=architecture_config.get("gru2_hidden", 96),
        dense1_hidden=architecture_config.get("dense1_hidden", 32),
        dense2_hidden=architecture_config.get("dense2_hidden", 12),
        device=device,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)  # Move model to device
    model.eval()
    
    return model, training_info


def inference_gru_tf(
    model: BatteryGRUWrapperTF,
    test_dict_list: Sequence[Dict[str, Any]],
    device: Union[str, torch.device] = "cpu",
    return_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Perform inference on test data using auto-regressive rollout.
    
    Args:
        model: BatteryGRUWrapperTF model (should be in eval mode)
        test_dict_list: List of test profiles (dict format)
        device: Device to run inference on
        return_predictions: If True, return predictions for each profile
    
    Returns:
        Dictionary containing:
            - avg_rmse_mV: Average RMSE across all profiles (mV)
            - rmse_list: List of RMSE for each profile (mV)
            - time_segmented_rmse: Dictionary of RMSE by time segments
            - predictions: (optional) List of prediction dictionaries for each profile
    """
    model.eval()
    device = torch.device(device)
    
    if not test_dict_list:
        return {"avg_rmse_mV": float("nan"), "rmse_list": [], "time_segmented_rmse": {}}
    
    print(f"[inference_gru_tf] Input test_dict_list contains {len(test_dict_list)} profiles")
    
    rmse_list: List[float] = []
    y_std_list: List[float] = []
    pred_all: List[np.ndarray] = []
    target_all: List[np.ndarray] = []
    time_all: List[np.ndarray] = []
    predictions_list: List[Dict[str, np.ndarray]] = []
    
    with torch.no_grad():
        for profile_idx, profile in enumerate(test_dict_list):
            print(f"[inference_gru_tf] Processing profile {profile_idx + 1}/{len(test_dict_list)}")
            prepared, feature_tensor, y_norm = _prepare_profile_tensors_tf(profile, device)
            total_steps = y_norm.shape[0]
            
            if total_steps <= 1:
                print(f"  [inference_gru_tf] Profile {profile_idx + 1}: Skipped (total_steps <= 1)")
                continue
            
            # Initialize predictions
            pred_y_norm = y_norm.clone()  # Start with true value at k=0
            features_autoreg = feature_tensor.clone()  # [T, 6]
            
            # Initialize hidden states
            gru1_hidden = None
            gru2_hidden = None
            
            # Auto-regressive rollout: k=0 to k=total_steps-2 (predict k+1)
            for k in range(total_steps - 1):
                # Get input at timestep k: [1, 1, 6]
                input_k = features_autoreg[k:k+1, :].unsqueeze(0)  # [1, 1, 6]
                # Use predicted value (auto-regressive)
                input_k[0, 0, 5] = pred_y_norm[k]  # Y(k) = predicted value
                
                # Forward pass
                if gru1_hidden is None:
                    preds_seq, (gru1_hidden, gru2_hidden) = model(input_k, hidden=None)
                else:
                    preds_seq, (gru1_hidden, gru2_hidden) = model(input_k, hidden=(gru1_hidden, gru2_hidden))
                
                # Get prediction: delta_Vcorr(k+1)
                pred_delta = preds_seq[0, 0, 0].item()
                
                # Update predicted Y(k+1)
                pred_y_norm[k + 1] = pred_y_norm[k] + pred_delta
                features_autoreg[k + 1, 5] = pred_y_norm[k + 1]
            
            # Calculate RMSE (exclude first timestep)
            if total_steps > 1:
                pred_valid = pred_y_norm[1:].cpu().numpy()
                target_valid = y_norm[1:].cpu().numpy()
                time_valid = np.array(prepared["time"][1:])  # Time array (exclude first timestep)
                
                # Denormalize predictions
                y_std = prepared["Y_std"]
                y_mean = prepared["Y_mean"]
                pred_denorm = pred_valid * y_std + y_mean
                target_denorm = target_valid * y_std + y_mean
                
                rmse_norm = float(np.sqrt(np.mean((pred_valid - target_valid) ** 2)))
                rmse_mV = rmse_norm * y_std * 1000
                
                rmse_list.append(rmse_mV)
                y_std_list.append(y_std)
                
                # Store for time-segmented RMSE calculation
                pred_all.append(pred_valid)
                target_all.append(target_valid)
                time_all.append(time_valid)
                
                # Store predictions if requested
                if return_predictions:
                    predictions_list.append({
                        "time": time_valid,
                        "pred_norm": pred_valid,
                        "target_norm": target_valid,
                        "pred_denorm": pred_denorm,
                        "target_denorm": target_denorm,
                        "rmse_mV": rmse_mV,
                    })
    
    avg_rmse_mV = float(np.nanmean(rmse_list)) if rmse_list else float("nan")
    
    # Calculate time-segmented RMSE
    time_segments = [30, 60, 120, 180, 300, 600, 900, 1200, float("inf")]
    time_segmented_rmse = {}
    
    if pred_all:
        pred_flat = np.concatenate(pred_all)
        target_flat = np.concatenate(target_all)
        time_flat = np.concatenate(time_all)
        
        for t_max in time_segments:
            mask = time_flat <= t_max
            if mask.any():
                rmse_seg = float(np.sqrt(np.mean((pred_flat[mask] - target_flat[mask]) ** 2)))
                avg_y_std = float(np.nanmean(y_std_list)) if y_std_list else 1.0
                rmse_seg_mV = rmse_seg * avg_y_std * 1000
                label = f"{int(t_max)}s" if t_max != float("inf") else "all"
                time_segmented_rmse[label] = rmse_seg_mV
        
        # Calculate rest RMSE (after 1730s)
        mask_rest = time_flat > 1730
        if mask_rest.any():
            rmse_rest = float(np.sqrt(np.mean((pred_flat[mask_rest] - target_flat[mask_rest]) ** 2)))
            avg_y_std = float(np.nanmean(y_std_list)) if y_std_list else 1.0
            rmse_rest_mV = rmse_rest * avg_y_std * 1000
            time_segmented_rmse["rest_after_1730s"] = rmse_rest_mV
    
    result = {
        "avg_rmse_mV": avg_rmse_mV,
        "rmse_list": rmse_list,
        "time_segmented_rmse": time_segmented_rmse,
    }
    
    if return_predictions:
        result["predictions"] = predictions_list
    
    return result


def plot_inference_results(
    results: Dict[str, Any],
    test_dict_list: Sequence[Dict[str, Any]],
    title_prefix: str = "GRU Inference Results",
) -> go.Figure:
    """
    Plot inference results: V_meas vs (V_spme + V_corr_pred_denormalized).
    
    Args:
        results: Results dictionary from inference_gru_tf
        test_dict_list: Original test profiles (list of dict)
        title_prefix: Prefix for plot titles
    
    Returns:
        Plotly figure object
    """
    if "predictions" not in results or not results["predictions"]:
        fig = go.Figure()
        fig.update_layout(title="No predictions to plot")
        return fig
    
    predictions = results["predictions"]
    num_profiles = len(predictions)
    
    print(f"[plot_inference_results] Plotting {num_profiles} profiles from results")
    print(f"[plot_inference_results] Input test_dict_list contains {len(test_dict_list)} profiles")
    
    # Validate input lengths match
    if len(test_dict_list) != num_profiles:
        print(f"[plot_inference_results] WARNING: predictions ({num_profiles}) and test_dict_list ({len(test_dict_list)}) lengths don't match")
        num_profiles = min(num_profiles, len(test_dict_list))
        predictions = predictions[:num_profiles]
        test_dict_list = test_dict_list[:num_profiles]
    
    # Check if we have any valid profiles to plot
    if num_profiles == 0:
        print(f"[plot_inference_results] No valid profiles to plot")
        fig = go.Figure()
        fig.update_layout(title="No valid profiles to plot")
        return fig
    
    # Calculate vertical_spacing dynamically based on number of rows
    # Maximum allowed spacing is 1 / (rows - 1), use 85% of that for safety margin
    if num_profiles > 1:
        max_vertical_spacing = 1.0 / (num_profiles - 1)
        vertical_spacing = min(0.08, max_vertical_spacing * 0.85)
    else:
        vertical_spacing = 0.3
    
    # Calculate reasonable height (max 8000px to avoid browser issues)
    base_height_per_profile = 250
    calculated_height = base_height_per_profile * num_profiles
    plot_height = min(calculated_height, 8000)
    
    if num_profiles > 30:
        print(f"[plot_inference_results] WARNING: {num_profiles} profiles may result in a very large plot. Consider filtering.")
    
    # Generate subplot titles
    subplot_titles = []
    for i in range(num_profiles):
        subplot_titles.append(f"Profile {i+1} - V_meas vs V_spme+V_corr_pred")
        subplot_titles.append(f"Profile {i+1} - Vcorr vs V_corr_pred")
    
    # Two subplots per profile: V_meas vs V_spme+V_corr_pred, and Vcorr vs V_corr_pred_denorm
    try:
        fig = make_subplots(
            rows=num_profiles,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.1,
            shared_xaxes=True,
        )
    except Exception as e:
        print(f"[plot_inference_results] Error creating subplots: {e}")
        # Fallback: try with smaller spacing if needed
        if num_profiles > 1:
            max_vertical_spacing = 1.0 / (num_profiles - 1)
            vertical_spacing = max_vertical_spacing * 0.8
        fig = make_subplots(
            rows=num_profiles,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.1,
            shared_xaxes=True,
        )
    
    # Track successful plots
    successful_plots = 0
    
    for idx, (pred_dict, profile) in enumerate(zip(predictions, test_dict_list)):
        try:
            print(f"[plot_inference_results] Plotting profile {idx + 1}/{num_profiles}")
            row = idx + 1
            
            # Get time data
            time_data = np.array(pred_dict.get("time", []))
            if len(time_data) == 0:
                print(f"  [plot_inference_results] Profile {idx + 1}: No time data, skipping")
                continue
            
            # Get V_meas from original profile (skip first timestep to match predictions)
            v_meas = None
            if "df" in profile:
                df = profile["df"]
                if "V_meas" in df.columns:
                    v_meas = df["V_meas"].values[1:]  # Skip first timestep to match predictions
            elif "V_meas" in profile:
                v_meas = np.array(profile["V_meas"])[1:]  # Skip first timestep
            
            # Get V_corr_pred_denormalized from predictions
            v_corr_pred_denorm = np.array(pred_dict.get("pred_denorm", []))
            if len(v_corr_pred_denorm) == 0:
                print(f"  [plot_inference_results] Profile {idx + 1}: No pred_denorm data, skipping")
                continue
            
            # Get V_spme from original profile
            v_spme = None
            if "df" in profile:
                df = profile["df"]
                if "V_spme" in df.columns:
                    v_spme = df["V_spme"].values[1:]  # Skip first timestep
            elif "V_spme" in profile:
                v_spme = np.array(profile["V_spme"])[1:]  # Skip first timestep
            
            # Get Vcorr (true value) from original profile
            vcorr = None
            if "df" in profile:
                df = profile["df"]
                if "Vcorr" in df.columns:
                    vcorr = df["Vcorr"].values[1:]  # Skip first timestep
            elif "Vcorr" in profile:
                vcorr = np.array(profile["Vcorr"])[1:]  # Skip first timestep
            
            # Skip if essential data is missing
            if v_meas is None or v_spme is None:
                print(f"  [plot_inference_results] Profile {idx + 1}: Skipped (V_meas or V_spme not found)")
                continue
            
            # Ensure same length for first subplot (V_meas vs V_spme+V_corr_pred)
            min_len1 = min(len(time_data), len(v_meas), len(v_spme), len(v_corr_pred_denorm))
            if min_len1 == 0:
                print(f"  [plot_inference_results] Profile {idx + 1}: No valid data points, skipping")
                continue
                
            time_data1 = time_data[:min_len1]
            v_meas_plot = v_meas[:min_len1]
            v_spme_plot = v_spme[:min_len1]
            v_corr_pred_denorm1 = v_corr_pred_denorm[:min_len1]
            
            # Calculate V_spme + V_corr_pred_denormalized
            v_spme_corr = v_spme_plot + v_corr_pred_denorm1
            
            # Plot 1: V_meas vs V_spme + V_corr_pred (left column)
            fig.add_trace(
                go.Scatter(
                    x=time_data1,
                    y=v_meas_plot,
                    mode="lines",
                    name="V_meas" if idx == 0 else None,
                    line=dict(color="blue", width=1.5),
                    legendgroup="v_meas",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=1,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_data1,
                    y=v_spme_corr,
                    mode="lines",
                    name="V_spme + V_corr_pred" if idx == 0 else None,
                    line=dict(color="red", width=1.5, dash="dash"),
                    legendgroup="v_spme_corr",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=1,
            )
            
            # Plot 2: Vcorr vs V_corr_pred_denorm (right column)
            if vcorr is not None and len(vcorr) > 0:
                min_len2 = min(len(time_data), len(vcorr), len(v_corr_pred_denorm))
                if min_len2 > 0:
                    time_data2 = time_data[:min_len2]
                    vcorr_plot = vcorr[:min_len2]
                    v_corr_pred_denorm2 = v_corr_pred_denorm[:min_len2]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_data2,
                            y=vcorr_plot,
                            mode="lines",
                            name="Vcorr (true)" if idx == 0 else None,
                            line=dict(color="green", width=1.5),
                            legendgroup="vcorr",
                            showlegend=(idx == 0),
                        ),
                        row=row,
                        col=2,
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_data2,
                            y=v_corr_pred_denorm2,
                            mode="lines",
                            name="V_corr_pred_denorm" if idx == 0 else None,
                            line=dict(color="orange", width=1.5, dash="dash"),
                            legendgroup="v_corr_pred_denorm",
                            showlegend=(idx == 0),
                        ),
                        row=row,
                        col=2,
                    )
            else:
                print(f"  [plot_inference_results] Profile {idx + 1}: Vcorr not found, skipping second subplot")
            
            # Update axes
            if idx == 0:
                fig.update_xaxes(title_text="Time (s)", row=row, col=1)
                fig.update_xaxes(title_text="Time (s)", row=row, col=2)
            fig.update_yaxes(title_text="Voltage (V)", row=row, col=1)
            fig.update_yaxes(title_text="Vcorr (V)", row=row, col=2)
            
            successful_plots += 1
            
        except Exception as e:
            print(f"  [plot_inference_results] Error plotting profile {idx + 1}: {e}")
            continue
    
    print(f"[plot_inference_results] Successfully plotted {successful_plots}/{num_profiles} profiles")
    
    avg_rmse = results.get("avg_rmse_mV", float("nan"))
    fig.update_layout(
        title=f"{title_prefix} | Total Profiles: {num_profiles} | Avg RMSE: {avg_rmse:.2f} mV",
        height=plot_height,
        width=1400,  # Wider to accommodate two columns
        hovermode="x unified",
    )
    
    return fig

