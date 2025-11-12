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
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class DNNBenchmark(nn.Module):
    """
    Feed-forward network that mirrors the Neural ODE driving head.
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM):
        super().__init__()
        self.net = _build_mlp(
            input_dim=input_dim,
            hidden_dims=(32, 32, 32, 16),
            output_dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., input_dim)
        Returns:
            (..., 1) predicted Vcorr(k+1)
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

    vcorr = _resolve_array(profile, "Vcorr")
    if vcorr is None:
        Y_norm = _resolve_array(profile, "Y")
        if Y_norm is not None and "Y_mean" in profile and "Y_std" in profile:
            vcorr = Y_norm * float(profile["Y_std"]) + float(profile["Y_mean"])
        elif v_meas is not None and v_spme is not None:
            vcorr = v_meas - v_spme
        else:
            raise KeyError(
                "Profile must contain Vcorr information: provide 'Vcorr', "
                "'Y' with stats, or both 'V_meas' and 'V_spme'."
            )

    if any(arr is None for arr in (V_ref, ocv, vcorr, soc, current, temperature, v_spme, v_meas)):
        missing = [
            name
            for name, arr in zip(
                ["V_ref/V_meas", "ocv", "Vcorr", "SOC", "I", "T", "V_spme", "V_meas"],
                (V_ref, ocv, vcorr, soc, current, temperature, v_spme, v_meas),
            )
            if arr is None
        ]
        raise KeyError(f"Missing required arrays: {missing}")

    min_len = min(
        len(V_ref), len(ocv), len(vcorr), len(soc), len(current), len(temperature), len(v_spme), len(v_meas)
    )
    if min_len < 2:
        raise ValueError("Each profile must contain at least two time steps.")

    features = np.stack(
        [
            V_ref[:min_len],
            ocv[:min_len],
            vcorr[:min_len],
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
    """

    def __init__(self, profiles: Sequence[dict]):
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        v_spme_next: List[np.ndarray] = []
        v_meas_next: List[np.ndarray] = []
        for profile in profiles:
            features, vcorr, v_spme, v_meas = _extract_features_and_targets(profile)
            xs.append(features[:-1])
            ys.append(vcorr[1:, None])
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
    dict_list: Sequence[dict],
    num_epochs: int,
    lr: float,
    device: Union[str, torch.device],
    training_batch_size: Optional[int],
    num_workers: int,
    *,
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
    """
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = _resolve_device(device)

    dataset = build_dnn_dataset(dict_list)
    effective_batch_size = training_batch_size if training_batch_size is not None else len(dataset)
    drop_last = training_batch_size is not None
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = DNNBenchmark().to(device)
    _load_pretrained_state(model, pretrained_model_path)

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

    best_rmse = float("inf")
    best_epoch = -1
    best_state_dict = None
    history: Dict[str, List[float]] = {
        "epoch": [],
        "loss": [],
        "rmse_mV": [],
        "lr": [],
        "grad_before": [],
        "grad_after": [],
    }
    rmse_window = deque(maxlen=early_stop_window)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        voltage_sse = 0.0
        sample_count = 0
        grad_norm_before = 0.0
        grad_norm_after = 0.0

        for xb, yb, v_spme_next, v_meas_next in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            v_spme_next = v_spme_next.to(device, non_blocking=True)
            v_meas_next = v_meas_next.to(device, non_blocking=True)

            preds = model(xb)
            loss = criterion(preds, yb)
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

            running_loss += loss.item() * xb.size(0)
            voltage_sse += torch.sum((v_spme_next + preds.detach() - v_meas_next) ** 2).item()
            sample_count += xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_rmse = (voltage_sse / sample_count) ** 0.5
        scheduler.step(epoch_loss)

        avg_grad_before = grad_norm_before / len(loader)
        avg_grad_after = grad_norm_after / len(loader)

        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        history["rmse_mV"].append(epoch_rmse * 1e3)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grad_before"].append(avg_grad_before)
        history["grad_after"].append(avg_grad_after)

        if verbose:
            print(
                f"[DNN] epoch {epoch}/{num_epochs}, "
                f"loss={epoch_loss:.3e}, "
                f"RMSE={epoch_rmse*1e3:.2f} mV, "
                f"LR={optimizer.param_groups[0]['lr']:.3e}, "
                f"Grad={avg_grad_before:.3e}->{avg_grad_after:.3e}"
            )

        rmse_window.append(epoch_rmse * 1e3)
        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"[DNN] ✅ Best RMSE improved to {best_rmse*1e3:.2f} mV at epoch {epoch}")

        if (
            rmse_window.maxlen == len(rmse_window)
            and (max(rmse_window) - min(rmse_window)) <= early_stop_delta_mV
        ):
            if verbose:
                print(f"[DNN] Early stopping triggered (ΔRMSE <= {early_stop_delta_mV:.3f} mV over {early_stop_window} epochs)")
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
        "batch_size": training_batch_size or len(loader.dataset),
        "num_workers": num_workers,
        "best_epoch": best_epoch,
        "best_rmse_mV": best_rmse * 1e3,
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
        batch_size=training_batch_size or len(loader.dataset),
        num_workers=num_workers,
        best_epoch=best_epoch,
        best_rmse_mV=best_rmse * 1e3,
        history=history,
    )

    if verbose:
        print(f"[DNN] Best RMSE = {summary.best_rmse_mV:.2f} mV at epoch {summary.best_epoch}")
        print(f"[DNN] Saved best model to {summary.state_dict_path}")

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
) -> np.ndarray:
    """
    Perform an autoregressive rollout over a single profile.

    Args:
        model: trained DNNBenchmark
        features: np.ndarray shape [T, 6] as returned by _extract_features_and_targets
        device: torch device
    Returns:
        np.ndarray shape [T] with predicted Vcorr in physical units (denormalized)
    """
    seq_len = features.shape[0]
    static_features = torch.from_numpy(features).to(device)
    pred_vcorr = torch.zeros(seq_len, dtype=torch.float32, device=device)
    pred_vcorr[0] = static_features[0, 2]

    with torch.no_grad():
        for k in range(seq_len - 1):
            input_vec = torch.stack(
                [
                    static_features[k, 0],  # V_ref or V_meas
                    static_features[k, 1],  # ocv
                    pred_vcorr[k],          # autoregressive Vcorr(k)
                    static_features[k, 3],  # SOC
                    static_features[k, 4],  # current
                    static_features[k, 5],  # temperature
                ]
            ).unsqueeze(0)
            pred_next = model(input_vec).squeeze(0).squeeze(0)
            pred_vcorr[k + 1] = pred_next

    return pred_vcorr.detach().cpu().numpy()


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
        pred_vcorr = _dnn_autoregressive_rollout(model, features, device)

        time_values = _resolve_array(profile, "time")
        soc_values = _resolve_array(profile, "SOC")
        profile_name = _resolve_profile_name(profile, fallback=f"profile_{idx}")

        v_pred = v_spme + pred_vcorr
        squared_error = np.sum((v_pred - v_meas) ** 2)
        rmse = float(np.sqrt(np.mean((v_pred - v_meas) ** 2)))

        total_squared_error += squared_error
        total_points += len(v_meas)

        results.append(
            {
                "profile_index": idx,
                "profile_name": profile_name,
                "time": time_values,
                "V_meas": v_meas,
                "V_spme": v_spme,
                "Vcorr_pred": pred_vcorr,
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


