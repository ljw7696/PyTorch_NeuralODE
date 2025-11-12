"""
Benchmark models for one-step-ahead Vcorr prediction.

Each model consumes the same feature vector that feeds the Neural ODE
(`V_ref`, `ocv`, `Vcorr`, `SOC`, `I`, `T`) at time step *k* and predicts
`Vcorr(k+1)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


