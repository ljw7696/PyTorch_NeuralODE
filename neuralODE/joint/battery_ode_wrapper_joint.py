import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchdiffeq import odeint
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import copy
from collections import deque
from pathlib import Path


class BatteryODEWrapperJoint(nn.Module):
    """
    Joint Neural ODE wrapper for battery voltage correction - supports both driving and rest profiles
    
    Architecture:
    - Driving head: 6 → 32 → 32 → 32 → 16 → 1
    - Rest head: 6 → 16 → 1
    - Gating layer: 3 → 8 → 1 (combines driving output, rest output, I_k)
    
    Inputs:
        t       : current time [s] 
        x       : state vector [1x1] = [Vcorr] (single) or [batch_size, 1] (batch)
        inputs  : dict with time-varying inputs (V_ref(t), ocv(t), SOC(t), I(t), T(t))
    Output:
        dxdt    : time derivative of state vector [1x1] = [dVcorr/dt] (single) or [batch_size, 1] (batch)
    """
    
    def __init__(self, device='cpu'):
        super(BatteryODEWrapperJoint, self).__init__()
        
        # Driving head: 6 → 32 → 32 → 32 → 16 → 1
        self.driving_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # Rest head: 6 → 16 → 1
        self.rest_net = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # Gating layer: I_k^2 → σ(w * I_k^2 + b)
        # Produces scalar gate to mix driving/rest outputs
        self.gating_layer = nn.Sequential(
            nn.Linear(1, 1),  # w*I² + b (2개 파라미터)
            nn.Sigmoid()       # σ(w*I² + b)
        )
        
        # Initialize weights
        # Driving head: will be overwritten by pretrained model if provided
        # Use smaller gain (0.1) for initialization in case pretrained model is not provided
        for m in self.driving_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
        # Rest head: will be overwritten by pretrained model if provided
        # Use smaller gain (0.1) for initialization in case pretrained model is not provided
        for m in self.rest_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
        # Gating layer: always trained from scratch, use moderate gain (0.5) for better learning
        # Combines pretrained driving and rest outputs, so moderate initialization is appropriate
        for m in self.gating_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
        
        self.device = device
        
        # Store interpolation functions for time-varying inputs
        self.inputs_interp = None
        
        # Batch processing variables
        self.inputs_batch = None
        self.t_common = None
        self.dt = None
        self.batch_indices = None
        
        # Initialize step_count for tracking
        self.step_count = 0
        
    def set_inputs(self, inputs_dict):
        """
        Set up interpolation functions for time-varying inputs
        Similar to MATLAB inputs struct with I(t), T(t), etc.
        
        Args:
            inputs_dict: {'time': [...], 'V_ref': [...], 'ocv': [...], 'SOC': [...], 'I': [...], 'T': [...], 'V_spme': [...]}
        """
        time_data = inputs_dict['time']
        
        self.inputs_interp = {
            'V_ref': interp1d(time_data, inputs_dict['V_ref'], kind='linear', fill_value='extrapolate'),
            'ocv': interp1d(time_data, inputs_dict['ocv'], kind='linear', fill_value='extrapolate'),
            'SOC': interp1d(time_data, inputs_dict['SOC'], kind='linear', fill_value='extrapolate'),
            'I': interp1d(time_data, inputs_dict['I'], kind='linear', fill_value='extrapolate'),
            'T': interp1d(time_data, inputs_dict['T'], kind='linear', fill_value='extrapolate'), 
            'V_spme': interp1d(time_data, inputs_dict['V_spme'], kind='linear', fill_value='extrapolate'),
        }
    
    def set_inputs_batch(self, inputs_list, t_common):
        """
        Set up pre-interpolated inputs for batch processing
        
        Args:
            inputs_list: List of dicts, each dict contains {'time', 'V_ref', 'ocv', 'SOC', 'I', 'T', 'V_spme'}
                        One dict per profile in the batch
            t_common: Common time vector [0, t_final] for all profiles (numpy array)
        """
        batch_size = len(inputs_list)
        num_timesteps = len(t_common)
        
        # Store common time information
        self.t_common = torch.tensor(t_common, dtype=torch.float32, device=self.device)
        self.dt = (t_common[-1] - t_common[0]) / (num_timesteps - 1) if num_timesteps > 1 else 1.0
        
        # Pre-interpolate all inputs for all profiles
        V_ref_batch = []
        ocv_batch = []
        SOC_batch = []
        I_batch = []
        T_batch = []
        V_spme_batch = []
        
        for profile_data in inputs_list:
            # Create interpolation for this profile
            time_orig = profile_data['time']
            
            V_ref_interp = interp1d(time_orig, profile_data['V_ref'], kind='linear', fill_value='extrapolate')
            ocv_interp = interp1d(time_orig, profile_data['ocv'], kind='linear', fill_value='extrapolate')
            SOC_interp = interp1d(time_orig, profile_data['SOC'], kind='linear', fill_value='extrapolate')
            I_interp = interp1d(time_orig, profile_data['I'], kind='linear', fill_value='extrapolate')
            T_interp = interp1d(time_orig, profile_data['T'], kind='linear', fill_value='extrapolate')
            V_spme_interp = interp1d(time_orig, profile_data['V_spme'], kind='linear', fill_value='extrapolate')
            
            # Evaluate at common time points
            V_ref_batch.append(V_ref_interp(t_common))
            ocv_batch.append(ocv_interp(t_common))
            SOC_batch.append(SOC_interp(t_common))
            I_batch.append(I_interp(t_common))
            T_batch.append(T_interp(t_common))
            V_spme_batch.append(V_spme_interp(t_common))
        
        # Convert to GPU tensors: shape (num_timesteps, batch_size)
        self.inputs_batch = {
            'V_ref': torch.tensor(np.array(V_ref_batch).T, dtype=torch.float32, device=self.device),
            'ocv': torch.tensor(np.array(ocv_batch).T, dtype=torch.float32, device=self.device),
            'SOC': torch.tensor(np.array(SOC_batch).T, dtype=torch.float32, device=self.device),
            'I': torch.tensor(np.array(I_batch).T, dtype=torch.float32, device=self.device),
            'T': torch.tensor(np.array(T_batch).T, dtype=torch.float32, device=self.device),
            'V_spme': torch.tensor(np.array(V_spme_batch).T, dtype=torch.float32, device=self.device),
        }
        
        print(f"✓ Batch inputs set: {batch_size} profiles, {num_timesteps} time steps")
    
    def forward(self, t, x):
        """
        ODE function called by odeint - supports both single and batch modes
        
        Joint Neural ODE: dVcorr/dt = gating(driving_net(...), rest_net(...), I)
        where driving_net and rest_net process the same 6 inputs independently,
        and gating layer combines their outputs with I
        
        Args:
            t: current time (scalar)
            x: state vector [Vcorr] (1x1 tensor for single, batch_size x 1 for batch)
            
        Returns:
            dxdt: derivative [dVcorr/dt] (1x1 tensor for single, batch_size x 1 for batch)
        """
        # Convert time to scalar if needed
        if isinstance(t, torch.Tensor):
            t_val = t.item()
        else:
            t_val = float(t)

        # Check if batch mode
        if self.inputs_batch is not None and self.t_common is not None:
            # Batch mode
            if isinstance(x, torch.Tensor):
                Vcorr_k = x[:, 0]  # (batch_size,)
            else:
                Vcorr_k = x
            
            # Find time index for direct tensor access
            t_idx = int((t_val - self.t_common[0].item()) / self.dt)
            t_idx = max(0, min(t_idx, len(self.t_common) - 1))
            
            # Get inputs at time t for all profiles - GPU tensor indexing
            if self.batch_indices is not None:
                V_ref_k = self.inputs_batch['V_ref'][t_idx, self.batch_indices]
                ocv_k = self.inputs_batch['ocv'][t_idx, self.batch_indices]
                SOC_k = self.inputs_batch['SOC'][t_idx, self.batch_indices]
                I_k = self.inputs_batch['I'][t_idx, self.batch_indices]
                T_k = self.inputs_batch['T'][t_idx, self.batch_indices]
            else:
                V_ref_k = self.inputs_batch['V_ref'][t_idx, :]
                ocv_k = self.inputs_batch['ocv'][t_idx, :]
                SOC_k = self.inputs_batch['SOC'][t_idx, :]
                I_k = self.inputs_batch['I'][t_idx, :]
                T_k = self.inputs_batch['T'][t_idx, :]
            
            # Neural Network Input: X = [V_ref(k), ocv(k), Vcorr(k), SOC(k), I(k), T(k)] for all profiles
            nn_input = torch.stack([V_ref_k, ocv_k, Vcorr_k, SOC_k, I_k, T_k], dim=1)  # (batch_size, 6)
            
            # Forward through both heads independently
            driving_output = self.driving_net(nn_input)  # (batch_size, 1)
            rest_output = self.rest_net(nn_input)        # (batch_size, 1)
            
            # Gating layer: computes gate from I_k^2
            gate = self.gating_layer(I_k.unsqueeze(1) ** 2)  # (batch_size, 1)
            dVcorr_dt_k = gate * driving_output + (1 - gate) * rest_output
            
        else:
            # Single profile mode (original)
            if isinstance(x, torch.Tensor):
                Vcorr_k = x[0, 0].item()
            else:
                Vcorr_k = float(x)

            # Evaluate time-varying inputs at time t
            if self.inputs_interp is None:
                raise ValueError("Must call set_inputs() before solving ODE")
                
            V_ref_k = float(self.inputs_interp['V_ref'](t_val))    # Vref(k)
            ocv_k = float(self.inputs_interp['ocv'](t_val))        # ocv(k)
            SOC_k = float(self.inputs_interp['SOC'](t_val))         # SOC(k)
            I_k = float(self.inputs_interp['I'](t_val))            # I(k)
            T_k = float(self.inputs_interp['T'](t_val))              # T(k)
            Vspme_k = float(self.inputs_interp['V_spme'](t_val))   # Vspme(k)
            
            # Neural Network Input: X = [Vref(k), ocv(k), Vcorr(k), SOC(k), I(k), T(k)]
            nn_input = torch.tensor([[V_ref_k, ocv_k, Vcorr_k, SOC_k, I_k, T_k]], dtype=torch.float32, device=self.device)
            
            # Forward through both heads independently
            driving_output = self.driving_net(nn_input)  # (1, 1)
            rest_output = self.rest_net(nn_input)        # (1, 1)
            
            # Gating layer: computes gate from I_k^2
            gate = self.gating_layer(torch.tensor([[I_k ** 2]], dtype=torch.float32, device=self.device))  # (1, 1)
            dVcorr_dt_k = gate * driving_output + (1 - gate) * rest_output

        self.step_count += 1
        
        # Return derivative for ODE solver
        return dVcorr_dt_k


def compute_grad_norm(model):
    """Compute gradient norm for a model"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_battery_neural_ode_joint(data_list, num_epochs=100, lr=1e-3, device='cpu', verbose=True, training_batch_size=None, ode_wrapper=None, method='euler', pretrained_driving_model_path=None, pretrained_rest_model_path=None, head_training=True, pretrained_joint_model=None):
    """
    Train battery Neural ODE on multiple profiles (batch version) - Joint architecture
    
    Args:
        data_list: List of data_dicts, each containing:
                  {'time', 'V_ref', 'ocv', 'SOC', 'I', 'T', 'V_spme', 'Y', 'Y_mean', 'Y_std'}
        num_epochs: number of training epochs
        lr: learning rate  
        device: device to use
        verbose: print progress
        training_batch_size: Number of profiles to use per training step (None = use all)
        ode_wrapper: Optional pre-existing BatteryODEWrapperJoint instance to continue training from
        method: ODE solver method
        pretrained_driving_model_path: Path to pretrained driving-only model checkpoint (.pth)
        pretrained_rest_model_path: Path to pretrained rest-only model checkpoint (.pth)
        head_training: If True, train driving/rest heads together with gating layer.
                      If False, freeze driving/rest heads and train only gating layer.
        
    Returns:
        ode_wrapper: trained ODE wrapper
        history: training history
    """
    import time
    start_time = time.time()
    
    total_profiles = len(data_list)
    training_batch_size = training_batch_size if training_batch_size is not None else total_profiles
    training_batch_size = min(training_batch_size, total_profiles)
    
    print(f"\n{'='*60}")
    print(f"Joint Batch Training: {total_profiles} total profiles")
    print(f"Training batch size: {training_batch_size} profiles per epoch")
    print(f"{'='*60}")
    
    # Normalize each profile's time to start at 0
    normalized_data_list = []
    for data in data_list:
        data_norm = data.copy()
        t_orig = np.array(data['time'])
        data_norm['time'] = t_orig - t_orig[0]
        normalized_data_list.append(data_norm)
    
    # Find common time range and length after normalization
    time_lengths = [len(data['time']) for data in normalized_data_list]
    time_durations = [data['time'][-1] for data in normalized_data_list]
    
    print(f"Profile lengths: {time_lengths}")
    print(f"Time durations: {[f'{d:.1f}s' for d in time_durations]}")
    
    max_length = max(time_lengths)
    max_duration = max(time_durations)
    
    # Create common time vector
    t_common = np.linspace(0, max_duration, max_length)
    print(f"Common time vector: [0, {max_duration:.1f}] with {max_length} points")
    
    # Prepare inputs for batch processing
    inputs_list = []
    targets_list = []
    Y_std_list = []
    
    for data in normalized_data_list:
        inputs_list.append({
            'time': data['time'],
            'V_ref': data['V_ref'],
            'ocv': data['ocv'],
            'SOC': data['SOC'],
            'I': data['I'],
            'T': data['T'],
            'V_spme': data['V_spme'],
        })
        targets_list.append(data['Y'])
        Y_std_list.append(data['Y_std'])
    
    # Allow passing a checkpoint path for the joint model
    if isinstance(pretrained_joint_model, (str, Path)):
        joint_model_path = str(pretrained_joint_model)
        checkpoint = torch.load(pretrained_joint_model, map_location=device, weights_only=False)
        model_instance = BatteryODEWrapperJoint(device).to(device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model_instance.load_state_dict(state_dict, strict=True)
        pretrained_joint_model = model_instance
        if verbose:
            print(f"\n✅ Loaded pretrained joint model from {joint_model_path}")

    # Create ODE wrapper or use provided one
    if ode_wrapper is None:
        ode_wrapper = BatteryODEWrapperJoint(device)
        ode_wrapper = ode_wrapper.to(device)
        
        # If a full pretrained joint model is provided, load it first
        if pretrained_joint_model is not None:
            ode_wrapper.load_state_dict(pretrained_joint_model.state_dict(), strict=True)
            if verbose:
                print("\n✅ Loaded pretrained joint model (all heads + gating layer)")
            # Prevent additional loading from separate paths
            pretrained_driving_model_path = None
            pretrained_rest_model_path = None

        # Load pretrained weights from separate driving and rest models if provided
        if pretrained_driving_model_path is not None or pretrained_rest_model_path is not None:
            if verbose:
                print("\n" + "="*60)
                print("Loading Pretrained Weights for Joint Training")
                print("="*60)
            
            # Load driving model
            if pretrained_driving_model_path is not None:
                if verbose:
                    print(f"\n📂 Loading driving model: {pretrained_driving_model_path}")
                try:
                    driving_checkpoint = torch.load(pretrained_driving_model_path, map_location=device, weights_only=False)
                    driving_state_dict = driving_checkpoint['model_state_dict'] if isinstance(driving_checkpoint, dict) and 'model_state_dict' in driving_checkpoint else driving_checkpoint
                    
                    # Map 'net.X' -> 'X' (remove 'net.' prefix for Sequential.load_state_dict)
                    # Sequential expects keys like '0.weight', '2.bias', etc. (not 'driving_net.0.weight')
                    driving_net_dict = {}
                    for key, value in driving_state_dict.items():
                        if key.startswith('net.'):
                            new_key = key.replace('net.', '')  # 'net.0.weight' -> '0.weight'
                            driving_net_dict[new_key] = value
                    
                    missing_driving, unexpected_driving = ode_wrapper.driving_net.load_state_dict(driving_net_dict, strict=False)
                    if verbose:
                        if len(missing_driving) == 0 and len(unexpected_driving) == 0:
                            print(f"✅ Driving net loaded successfully: {len(driving_net_dict)} weights matched perfectly!")
                        else:
                            print(f"⚠️  Driving net loaded with warnings: {len(driving_net_dict)} weights")
                            if missing_driving:
                                print(f"   Missing keys: {len(missing_driving)}")
                                if len(missing_driving) <= 10:
                                    for k in missing_driving:
                                        print(f"     - {k}")
                            if unexpected_driving:
                                print(f"   Unexpected keys: {len(unexpected_driving)}")
                                if len(unexpected_driving) <= 10:
                                    for k in unexpected_driving:
                                        print(f"     - {k}")
                    
                    # Print training info if available
                    if isinstance(driving_checkpoint, dict) and 'training_info' in driving_checkpoint:
                        training_info = driving_checkpoint['training_info']
                        if verbose:
                            print(f"   Training info: Best RMSE = {training_info.get('best_rmse', 'N/A'):.2f}mV at epoch {training_info.get('best_epoch', 'N/A')}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not load driving model: {e}")
            
            # Load rest model
            if pretrained_rest_model_path is not None:
                if verbose:
                    print(f"\n📂 Loading rest model: {pretrained_rest_model_path}")
                try:
                    rest_checkpoint = torch.load(pretrained_rest_model_path, map_location=device, weights_only=False)
                    rest_state_dict = rest_checkpoint['model_state_dict'] if isinstance(rest_checkpoint, dict) and 'model_state_dict' in rest_checkpoint else rest_checkpoint
                    
                    # Map 'net.X' -> 'X' (remove 'net.' prefix for Sequential.load_state_dict)
                    # Sequential expects keys like '0.weight', '2.bias', etc. (not 'rest_net.0.weight')
                    rest_net_dict = {}
                    for key, value in rest_state_dict.items():
                        if key.startswith('net.'):
                            new_key = key.replace('net.', '')  # 'net.0.weight' -> '0.weight'
                            rest_net_dict[new_key] = value
                    
                    missing_rest, unexpected_rest = ode_wrapper.rest_net.load_state_dict(rest_net_dict, strict=False)
                    if verbose:
                        if len(missing_rest) == 0 and len(unexpected_rest) == 0:
                            print(f"✅ Rest net loaded successfully: {len(rest_net_dict)} weights matched perfectly!")
                        else:
                            print(f"⚠️  Rest net loaded with warnings: {len(rest_net_dict)} weights")
                            if missing_rest:
                                print(f"   Missing keys: {len(missing_rest)}")
                                if len(missing_rest) <= 10:
                                    for k in missing_rest:
                                        print(f"     - {k}")
                            if unexpected_rest:
                                print(f"   Unexpected keys: {len(unexpected_rest)}")
                                if len(unexpected_rest) <= 10:
                                    for k in unexpected_rest:
                                        print(f"     - {k}")
                    
                    # Print training info if available
                    if isinstance(rest_checkpoint, dict) and 'training_info' in rest_checkpoint:
                        training_info = rest_checkpoint['training_info']
                        if verbose:
                            print(f"   Training info: Best RMSE = {training_info.get('best_rmse', 'N/A'):.2f}mV at epoch {training_info.get('best_epoch', 'N/A')}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not load rest model: {e}")
            
            if verbose:
                print("="*60)
                if head_training:
                    print("Note: All networks will be trained together (joint fine-tuning)")
                else:
                    print("Note: Gating layer will be trained from scratch (heads frozen)")
                print("="*60 + "\n")
    else:
        # Use provided wrapper and ensure it's on the correct device
        ode_wrapper = ode_wrapper.to(device)
        print(f"🔄 Using provided ODE wrapper for continued training")
    
    # Set batch inputs
    ode_wrapper.set_inputs_batch(inputs_list, t_common)
    
    # Prepare targets as tensors with padding
    targets_padded = []
    valid_lengths = []
    for target in targets_list:
        original_length = len(target)
        valid_lengths.append(original_length)
        if len(target) < max_length:
            padded = np.concatenate([target, np.full(max_length - len(target), target[-1])])
        else:
            padded = target[:max_length]
        targets_padded.append(padded)
    
    targets_batch = torch.tensor(np.array(targets_padded), dtype=torch.float32, device=device)
    
    # Create mask for valid (non-padded) regions
    valid_mask = torch.zeros((total_profiles, max_length), dtype=torch.bool, device=device)
    for i, valid_len in enumerate(valid_lengths):
        valid_mask[i, :valid_len] = True
    
    # Print Network Architecture
    print("\n" + "="*60)
    print("Neural Network Architecture (Joint)")
    print("="*60)
    print("Driving Network:")
    print(ode_wrapper.driving_net)
    print("\nRest Network:")
    print(ode_wrapper.rest_net)
    print("\nGating Layer:")
    print(ode_wrapper.gating_layer)
    print("="*60 + "\n")
    
    # Freeze driving/rest heads if head_training is False
    if not head_training:
        driving_frozen_count = 0
        rest_frozen_count = 0
        for param in ode_wrapper.driving_net.parameters():
            param.requires_grad = False
            driving_frozen_count += 1
        for param in ode_wrapper.rest_net.parameters():
            param.requires_grad = False
            rest_frozen_count += 1
        if verbose:
            print("="*60)
            print("⚠️  Head Training Mode: DISABLED")
            print(f"   - Driving head: FROZEN ({driving_frozen_count} parameters, pretrained weights preserved)")
            print(f"   - Rest head: FROZEN ({rest_frozen_count} parameters, pretrained weights preserved)")
            print("   - Gating layer: TRAINABLE")
            print("="*60 + "\n")
    else:
        if verbose:
            print("="*60)
            print("✓ Head Training Mode: ENABLED")
            print("   - Driving head: TRAINABLE")
            print("   - Rest head: TRAINABLE")
            print("   - Gating layer: TRAINABLE")
            print("="*60 + "\n")
    
    # Optimizer: only include parameters that require gradients
    trainable_params = [p for p in ode_wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, eps=1e-8, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-10,
        threshold=1e-3,
        threshold_mode='rel'
    )
    
    history = {
        'loss': [],
        'rmse': [],
        'loss_V': [],
        'loss_dVdt': [],
        'grad_norm_before': [],
        'grad_norm_after': [],
    }
    
    # Best model tracking
    best_rmse = float('inf')
    best_model_state = None
    best_epoch = 0
    
    epoch_window = 20
    rmse_hist = deque(maxlen=epoch_window)
    
    # Parameters
    alpha = 1
    beta = 0
    gamma = 0
    gpu_mem = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if device == 'cuda' else "N/A"
    grad_clip_max = 50
    
    # Use average Y_std for normalization display
    Y_std_avg = np.mean(Y_std_list)
    
    # Print Settings
    print("="*60)
    print("Training Settings")
    print("="*60)
    print(f"Total profiles: {total_profiles}")
    print(f"Training batch size: {training_batch_size}")
    print(f"Common time points: {max_length}")
    print(f"Max epochs: {num_epochs}")
    print(f"Initial LR: {lr}")
    print(f"Device: {device}")
    print(f"Patience: {scheduler.patience}")
    print(f"Alpha: {alpha:.2f}, Beta: {beta:.2f}, Gamma: {gamma:.2f}")
    print(f"GPU: {gpu_mem}")
    print(f"Grad clip max: {grad_clip_max}")
    print(f"Method: {method}")
    print(f"Verbose: {verbose}")
    print("="*60)
    
    if verbose:
        print("Starting joint batch training...")
        print(f"Targets shape: {targets_batch.shape}")
        print(f"Target range: {targets_batch.min():.3f} ~ {targets_batch.max():.3f}")
    
    t_eval = torch.tensor(t_common, dtype=torch.float32, device=device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Sample random batch if training_batch_size < total_profiles
        if training_batch_size < total_profiles:
            selected_indices = np.random.choice(total_profiles, training_batch_size, replace=False)
            batch_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
        else:
            batch_indices = None
        
        # Set batch indices for forward pass
        ode_wrapper.batch_indices = batch_indices
        
        # Prepare initial conditions
        if batch_indices is not None:
            x0 = targets_batch[batch_indices, 0:1]
        else:
            x0 = targets_batch[:, 0:1]
        
        # ODE solving - BATCH!
        ode_wrapper.step_count = 0
        solution = odeint(ode_wrapper, x0, t_eval, method=method)
        Vcorr_pred = solution[:, :, 0].T
        
        # Reset batch indices
        ode_wrapper.batch_indices = None
        
        # Calculate loss
        if batch_indices is not None:
            targets_selected = targets_batch[batch_indices]
            valid_mask_selected = valid_mask[batch_indices]
        else:
            targets_selected = targets_batch
            valid_mask_selected = valid_mask
        
        # Apply mask to exclude padded regions from loss
        Vcorr_pred_masked = Vcorr_pred * valid_mask_selected.float()
        targets_masked = targets_selected * valid_mask_selected.float()
        
        # Loss calculation
        loss_V = torch.tensor(0.0, device=device)
        num_valid = 0
        
        for i in range(len(targets_selected)):
            profile_valid_mask = valid_mask_selected[i, :].float()
            num_valid_i = profile_valid_mask.sum()
            if num_valid_i > 0:
                loss_V += torch.sum((Vcorr_pred_masked[i, :] - targets_masked[i, :]) ** 2)
                num_valid += num_valid_i
        
        if num_valid > 0:
            loss_V = loss_V / num_valid
        
        dVdt_pred = torch.diff(Vcorr_pred, dim=1) / torch.diff(t_eval).unsqueeze(0)
        dVdt_ref = torch.diff(targets_selected, dim=1) / torch.diff(t_eval).unsqueeze(0)
        
        valid_mask_diff = valid_mask_selected[:, :-1]
        
        loss_dVdt = torch.tensor(0.0, device=device)
        num_valid_diff = 0
        
        for i in range(len(targets_selected)):
            profile_valid_mask_diff = valid_mask_diff[i, :].float()
            num_valid_diff_i = profile_valid_mask_diff.sum()
            if num_valid_diff_i > 0:
                loss_dVdt += torch.sum((dVdt_pred[i, :] - dVdt_ref[i, :]) ** 2 * profile_valid_mask_diff)
                num_valid_diff += num_valid_diff_i
        
        if num_valid_diff > 0:
            loss_dVdt = loss_dVdt / num_valid_diff
        
        # Total loss
        loss_tv = torch.tensor(0.0, device=device)
        loss = alpha * loss_V + beta * loss_dVdt + gamma * loss_tv
        
        # Backward
        loss.backward()
        grad_norm_before = compute_grad_norm(ode_wrapper)
        torch.nn.utils.clip_grad_norm_(ode_wrapper.parameters(), max_norm=grad_clip_max)
        grad_norm_after = compute_grad_norm(ode_wrapper)
        
        history['grad_norm_before'].append(grad_norm_before)
        history['grad_norm_after'].append(grad_norm_after)
        
        optimizer.step()
        
        # History
        rmse = torch.sqrt(loss_V).item()
        history['loss'].append(loss.item())
        history['loss_dVdt'].append(loss_dVdt.item())
        history['loss_V'].append(loss_V.item())
        history['rmse'].append(rmse)
        
        # Save best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_model_state = copy.deepcopy(ode_wrapper.state_dict())
            print(f"✅ Best RMSE: {best_rmse * Y_std_avg * 1000:.2f}mV at Epoch {best_epoch+1}")
        
        # Update learning rate
        scheduler.step(loss_V)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print every epoch
        if verbose:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Method: {method:6s} | "
                  f"LR: {new_lr:.2e} | "
                  f"RMSE: {rmse * Y_std_avg * 1000:.2f}mV | "
                  f"loss_V: {loss_V.item():.4f} | "
                  f"loss_dVdt: {loss_dVdt.item():.2e} | "
                  f"Steps: {ode_wrapper.step_count:4d} | "
                  f"Grad: {grad_norm_before:.2f} → {grad_norm_after:.2f}")
        
        # Early stopping
        rmse_hist.append(rmse * Y_std_avg * 1000)
        if len(rmse_hist) >= epoch_window:
            if (max(rmse_hist) - min(rmse_hist)) <= 0.005:
                print("Window range <= 0.005 mV → stop")
                break
    
    # Load best model
    if best_model_state is not None:
        ode_wrapper.load_state_dict(best_model_state)
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Loaded best model from epoch {best_epoch+1}/{num_epochs}")
        print(f"Best RMSE: {best_rmse * Y_std_avg * 1000:.2f}mV")
        best_model_path = f"best_model_joint_rmse{best_rmse * Y_std_avg * 1000:.2f}mV.pth"
        
        # Calculate total training time
        total_training_time = time.time() - start_time
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = int(total_training_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        checkpoint = {
            'model_state_dict': best_model_state,
            'training_info': {
                'best_rmse': best_rmse * Y_std_avg * 1000,
                'best_epoch': best_epoch + 1,
                'total_epochs': num_epochs,
                'total_profiles': total_profiles,
                'training_batch_size': training_batch_size,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'initial_lr': lr,
                'max_epochs': num_epochs,
                'patience': scheduler.patience,
                'grad_clip_max': grad_clip_max,
                'method': method,
                'total_training_time_seconds': total_training_time,
                'total_training_time_formatted': time_str,
            },
        }
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to: {best_model_path}")
        print(f"Total training time: {time_str} ({total_training_time:.1f} seconds)")
        print(f"{'='*60}\n")
    
    return ode_wrapper, history
