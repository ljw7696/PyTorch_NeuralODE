import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchdiffeq import odeint_adjoint as odeint
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import copy
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class BatteryODEWrapper(nn.Module):
    """
    Neural ODE wrapper for battery voltage correction - mirrors MATLAB ode_wrapper_poly
    
    This replaces the physics-based solid_dynamics_poly and electrolyte_dynamics_poly
    with a neural network that learns dVcorr/dt from battery states.
    
    Inputs:
        t       : current time [s] 
        x       : state vector [1x1] = [Vcorr]
        inputs  : dict with time-varying inputs (V_ref(t), ocv(t), SOC(t), I(t), T(t))
        params  : neural network parameters (self.net)
    Output:
        dxdt    : time derivative of state vector [1x1] = [dVcorr/dt]
    """
    
    def __init__(self, device='cpu'):
        super(BatteryODEWrapper, self).__init__()
        
        # Neural network: input [V_ref(t), ocv(t), Vcorr(t), SOC(t), I(t), T(t)] -> output dVcorr/dt
        # 6 inputs: [V_ref_k, ocv_k, Vcorr_k, SOC_k, I_k, T_k]
        # 7 layers: original working version
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
            # nn.Linear(6, 16),
            # nn.Tanh(),
            # nn.Linear(16, 1)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1.0)
                # nn.init.constant_(m.bias, val=0)
                nn.init.orthogonal_(m.weight, gain=0.7)  # Orthogonal!
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
        
        # # Initialize prev_Vcorr for state tracking
        # self.prev_Vcorr = None
        
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
        
        # # Reset prev_Vcorr when inputs change
        # self.prev_Vcorr = None
    
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
        
        Neural ODE: dVcorr/dt = f(V_ref, ocv, Vcorr, SOC, I, T)
        where f() is the neural network
        
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
            
            # Neural Network Output: dVcorr/dt(k)
            dVcorr_dt_k = self.net(nn_input)
            
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
            
            # Neural Network Output: dVcorr/dt(k)
            dVcorr_dt_k = self.net(nn_input)

        self.step_count += 1
        
        # Return derivative for ODE solver
        return dVcorr_dt_k




def train_battery_neural_ode(data_dict, num_epochs=100, lr=1e-3, device='cpu', verbose=True, pretrained_model_path=None):
    """
    Train battery Neural ODE - mirrors MATLAB ode15s usage
    
    Args:
        data_dict: {'time', 'V_ref', 'ocv', 'SOC', 'I', 'T', 'V_spme', 'V_meas'}
        num_epochs: number of training epochs
        lr: learning rate  
        device: device to use
        verbose: print prsogress
        
    Returns:
        ode_wrapper: trained ODE wrapper
        history: training history
    """
    
    # Convert data to tensors
    time_data = torch.tensor(data_dict['time'], dtype=torch.float32, device=device)
    # csn_data = torch.tensor(data_dict['csn_bulk'], dtype=torch.float32, device=device)
    # I_data = torch.tensor(data_dict['I'], dtype=torch.float32, device=device)
    # T_data = torch.tensor(data_dict['T'], dtype=torch.float32, device=device)
    # V_spme = torch.tensor(data_dict['V_spme'], dtype=torch.float32, device=device)
    # V_meas = torch.tensor(data_dict['V_meas'], dtype=torch.float32, device=device)
    
    # Calculate target Vcorr
    Vcorr_target = torch.tensor(data_dict['Y'], dtype=torch.float32, device=device)  # CUDA tensor로 변환
    Ystd = data_dict['Y_std']

    # Create ODE wrapper (replaces ode_wrapper_poly)
    ode_wrapper = BatteryODEWrapper(device)
    ode_wrapper = ode_wrapper.to(device) # Move to GPU

    # Load pretrained model if provided
    if pretrained_model_path is not None:
        print(f"\n🔄 Loading pretrained model from: {pretrained_model_path}")
        try:
            loaded_data = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            
            # Check if it's a checkpoint (dict with 'model_state_dict') or just state_dict
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                # Checkpoint format - extract state_dict and print training info
                pretrained_state = loaded_data['model_state_dict']
                training_info = loaded_data.get('training_info', {})
                print(f"✅ Pretrained model loaded successfully!")
                if training_info:
                    print(f"   Training info from checkpoint:")
                    print(f"   - Best RMSE: {training_info.get('best_rmse', 'N/A'):.2f}mV")
                    print(f"   - Best epoch: {training_info.get('best_epoch', 'N/A')}")
                    print(f"   - alpha={training_info.get('alpha', 'N/A')}, "
                          f"beta={training_info.get('beta', 'N/A')}, "
                          f"gamma={training_info.get('gamma', 'N/A')}")
            else:
                # Legacy format - just state_dict
                pretrained_state = loaded_data
                print(f"✅ Pretrained model loaded successfully! (legacy format)")
            
            ode_wrapper.load_state_dict(pretrained_state)
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained model: {e}")
            print(f"   Starting from scratch...")
    # ...
    
    # Print Network Architecture
    print("\n" + "="*50)
    print("Neural Network Architecture")
    print("="*50)
    print(ode_wrapper.net)
    print("="*50 + "\n")
    
    # Set up inputs (similar to MATLAB inputs struct)
    ode_wrapper.set_inputs(data_dict)

    
    # optimizer = optim.Adam(ode_wrapper.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(ode_wrapper.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)
    # optimizer = optim.SGD(ode_wrapper.parameters(), lr=lr, momentum=0.9)
    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-10,
        threshold=1e-3,      # 1% 개선
        threshold_mode='rel'
        )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=20,      # 50 epoch마다 restart
    #     T_mult=2,
    #     eta_min=1e-6,
    # )

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=0.98  # 매 epoch마다 2% 감소
    # )

    
    history = {
    'loss': [], 
    'rmse': [], 
    'loss_V': [],      # 추가
    'loss_dVdt': [],    # 추가
    'grad_norm_before': [],
    'grad_norm_after': [],
    }

    # Best model tracking 초기화 (추가!)
    best_rmse = float('inf')
    best_model_state = None
    best_epoch = 0

    epoch_window = 20
    rmse_hist = deque(maxlen=epoch_window)
    
    # Other parameteres
    alpha = 1
    beta = 0
    gamma = 0
    gpu_mem = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if device == 'cuda' else "N/A"
    grad_clip_max = 50

    # Print Settings
    print("="*50)
    print("Training Settings")
    print("="*50)
    print(f"data_dict keys: {data_dict.keys()}")
    print(f"max epochs: {num_epochs}")
    print(f"initial lr: {lr}")
    print(f"device: {device}")
    print(f"patience: {scheduler.patience:2d}")
    print(f"alpha: {alpha:.2f}")
    print(f"beta: {beta:.2f}")
    print(f"gamma: {gamma:.2f}")
    print(f"GPU: {gpu_mem}")
    print(f"grad_clip_max: {grad_clip_max}")
    print(f"verbose: {verbose}")
    print("="*50)
    
    
    if verbose:
        print("Starting battery Neural ODE training...")
        print(f"Data points: {len(time_data)}")
        print(f"Vcorr range (normalized): {Vcorr_target.min():.3f} ~ {Vcorr_target.max():.3f}")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Solve ODE
        t_eval = torch.tensor(np.array(data_dict['time']), dtype=torch.float32, device=device)
        x0 = torch.tensor([[Vcorr_target[0].item()]], dtype=torch.float32, device=device)
        
        # ODE solving
        method = 'euler'
        ode_wrapper.step_count = 0
        solution = odeint(ode_wrapper, x0, t_eval, method=method)
        Vcorr_pred = solution[:, 0, 0]
        
        # Calculate loss
        dVdt_pred = torch.diff(Vcorr_pred) / torch.diff(t_eval)
        dVdt_ref = torch.diff(Vcorr_target) / torch.diff(t_eval)
        loss_dVdt = torch.mean((dVdt_pred - dVdt_ref ) ** 2) # 기울기의 차이를 최소화
        loss_V = torch.mean((Vcorr_pred - Vcorr_target) ** 2) # 전압의 차이를 최소화
        loss_tv = torch.mean(torch.abs(torch.diff(Vcorr_pred))) # 전압의 변화량을 최소화
        mae_V = torch.mean(torch.abs(Vcorr_pred - Vcorr_target))

        loss = alpha * loss_V + beta * loss_dVdt + gamma * loss_tv

        
        # Backprop
        loss.backward()
        grad_norm_before = compute_grad_norm(ode_wrapper)
        torch.nn.utils.clip_grad_norm_(ode_wrapper.parameters(), max_norm= grad_clip_max)
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
            print(f"✅ Best RMSE: {best_rmse * Ystd * 1000:.2f}mV at Epoch {best_epoch+1}")  # ← 수정!

        # Update learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss_V)  # ReduceLROnPlateau requires metric
        else:
            scheduler.step()  # Other schedulers (CosineAnnealingWarmRestarts, ExponentialLR, etc.)
        new_lr = optimizer.param_groups[0]['lr']
                
        # Print every epoch
        if verbose:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Method: {method:6s} | "
                f"LR: {new_lr:.2e} | "
                f"RMSE: {rmse * Ystd * 1000:.2f}mV | "
                f"loss_V: {loss_V.item():.2f} | "
                f"loss_dVdt: {loss_dVdt.item():.2e} | "
                f"loss_tv: {loss_tv.item():.2e} | "
                f"Steps: {ode_wrapper.step_count:4d} | "
                f"Grad Norm before: {grad_norm_before:.4f} | "
                f"Grad Norm after: {grad_norm_after:.4f} | "
                )


        # # Early Stopping
        # if rmse * Ystd * 1000 <= 1:
        #     print(f"RMSE is less than 1 mV, stopping training")
        #     break

        rmse_hist.append(rmse * Ystd * 1000)
        if len(rmse_hist) >= epoch_window-1: 
            if (max(rmse_hist) - min(rmse_hist)) <= 0.005:
                print("Window range <= 0.01 mV → stop")
                break

    # Load best model
    if best_model_state is not None:
        ode_wrapper.load_state_dict(best_model_state)
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Loaded best model from epoch {best_epoch+1}/{num_epochs}")
        print(f"Best RMSE: {best_rmse * Ystd * 1000:.2f}mV")
        best_model_path = f"best_model_rmse{best_rmse * Ystd * 1000:.2f}mV.pth"
        
        # Save checkpoint with important metadata
        # Get optimizer and scheduler type names dynamically
        optimizer_type = type(optimizer).__name__
        scheduler_type = type(scheduler).__name__
        
        # Get scheduler parameters (varies by scheduler type)
        scheduler_params = {}
        if hasattr(scheduler, 'gamma'):
            scheduler_params['gamma'] = scheduler.gamma
        if hasattr(scheduler, 'patience'):
            scheduler_params['patience'] = scheduler.patience
        if hasattr(scheduler, 'factor'):
            scheduler_params['factor'] = scheduler.factor
        if hasattr(scheduler, 'min_lr'):
            scheduler_params['min_lr'] = scheduler.min_lr
        if hasattr(scheduler, 'T_0'):
            scheduler_params['T_0'] = scheduler.T_0
        if hasattr(scheduler, 'eta_min'):
            scheduler_params['eta_min'] = scheduler.eta_min
        
        checkpoint = {
            'model_state_dict': best_model_state,
            'training_info': {
                'best_rmse': best_rmse * Ystd * 1000,  # mV
                'best_rmse_normalized': best_rmse,
                'best_epoch': best_epoch + 1,
                'total_epochs': num_epochs,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'initial_lr': lr,
                'optimizer_type': optimizer_type,
                'weight_decay': optimizer.param_groups[0]['weight_decay'],
                'scheduler_type': scheduler_type,
                'scheduler_params': scheduler_params,
                'grad_clip_max': grad_clip_max,
                'method': method,
                'device': device,
                'Y_std': Ystd,
                'Y_mean': data_dict.get('Y_mean', None),
            },
            'network_architecture': str(ode_wrapper.net)
        }
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to: {best_model_path}")
        print(f"  - Model weights")
        print(f"  - Training info (alpha={alpha}, beta={beta}, gamma={gamma})")
        print(f"  - Best RMSE: {best_rmse * Ystd * 1000:.2f}mV at epoch {best_epoch+1}")
        print(f"{'='*50}\n")

    else:
        print(f"\nWarning: No best model saved!\n")

    return ode_wrapper, history


def simulate_battery_ode(ode_wrapper, data_dict, device='cuda', plot=True):
    """
    트레이닝된 모델로 시뮬레이션 실행
    
    Args:
        ode_wrapper: 트레이닝된 BatteryODEWrapper 모델
        data_dict: {'time': [...], 'V_ref': [...], 'ocv': [...], 'SOC': [...], 'I': [...], 'T': [...], 
                   'Y_mean': float, 'Y_std': float, 'V_spme': [...]}
        device: device to use
        plot: plot results
    
    Returns:
        Vcorr_pred: 예측된 Vcorr (원본 스케일)
        Vcorr_target: 실제 Vcorr (원본 스케일)
        Vtotal_pred: 예측된 Vtotal = Vspme + Vcorr_pred
        Vtotal_meas: 실제 Vtotal = Vref
        t_eval: 시간 벡터
    """
    # Set to evaluation mode
    ode_wrapper.eval()
    
    # Clear batch mode if it was used during training
    ode_wrapper.inputs_batch = None
    ode_wrapper.t_common = None
    ode_wrapper.batch_indices = None
    
    # Set inputs for interpolation functions
    ode_wrapper.set_inputs(data_dict)
    
    # Reset state
    # ode_wrapper.prev_Vcorr = None
    
    # Prepare data
    t_eval = torch.tensor(np.array(data_dict['time']), dtype=torch.float32, device=device)
    
    # Get initial condition (normalized)
    if 'Y' in data_dict:
        x0_norm = torch.tensor([[data_dict['Y'][0]]], dtype=torch.float32, device=device)
    else:
        raise ValueError("data_dict must contain 'Y' for initial condition")
    
    # Run simulation (use faster solver for plotting)
    with torch.no_grad():
        method = 'euler'
        solution = odeint(ode_wrapper, x0_norm, t_eval, method=method)
        Vcorr_pred_norm = solution[:, 0, 0].cpu().numpy()
    
    # Denormalize
    if 'Y_mean' in data_dict and 'Y_std' in data_dict:
        Y_mean = data_dict['Y_mean']
        Y_std = data_dict['Y_std']
        Vcorr_pred = Vcorr_pred_norm * Y_std + Y_mean
    else:
        Vcorr_pred = Vcorr_pred_norm
        print("Warning: No Y_mean/Y_std found, returning normalized values")
    
    # Get target (if available)
    if 'Y' in data_dict:
        Y_target_norm = np.array(data_dict['Y'])
        if 'Y_mean' in data_dict and 'Y_std' in data_dict:
            Vcorr_target = Y_target_norm * Y_std + Y_mean
        else:
            Vcorr_target = Y_target_norm
    else:
        Vcorr_target = None
    
    # Calculate Vtotal
    if 'V_spme' in data_dict:
        V_spme = np.array(data_dict['V_spme'])
        
        Vtotal_pred = V_spme + Vcorr_pred
    else:
        Vtotal_pred = None
        print("Warning: No V_spme found, cannot calculate Vtotal")
    
    if 'V_meas' in data_dict:
        Vtotal_meas = np.array(data_dict['V_meas'])
    else:
        Vtotal_meas = None
    
    # Calculate RMSE
    rmse_vcorr = None
    rmse_vtotal = None
    
    if Vcorr_target is not None:
        rmse_vcorr = np.sqrt(np.mean((Vcorr_pred - Vcorr_target)**2)) * 1000  # mV
        print(f"\n{'='*50}")
        print(f"Simulation Results:")
        print(f"{'='*50}")
        print(f"Vcorr RMSE: {rmse_vcorr:.2f} mV")
    
    if Vtotal_meas is not None and Vtotal_pred is not None:
        rmse_vtotal = np.sqrt(np.mean((Vtotal_pred - Vtotal_meas)**2)) * 1000  # mV
        print(f"Vtotal RMSE: {rmse_vtotal:.2f} mV")
        print(f"{'='*50}\n")
    
    # Plot
    if plot:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        t_eval_np = t_eval.cpu().numpy()
        
        # Create subplots with secondary y-axis
        # Update subplot titles with RMSE
        subplot_title_vcorr = f'Vcorr Comparison'
        subplot_title_vtotal = f'Vtotal Comparison'
        
        if rmse_vcorr is not None:
            subplot_title_vcorr = f'Vcorr Comparison (RMSE: {rmse_vcorr:.2f} mV)'
        if rmse_vtotal is not None:
            subplot_title_vtotal = f'Vtotal Comparison (RMSE: {rmse_vtotal:.2f} mV)'
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(subplot_title_vcorr, subplot_title_vtotal),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]  # Enable secondary y-axis for both
        )
        
        # Vcorr comparison (top plot)
        if Vcorr_target is not None:
            fig.add_trace(
                go.Scatter(x=t_eval_np, y=Vcorr_target, 
                          name='Vcorr (measured)', 
                          line=dict(width=2), 
                          opacity=0.7),
                row=1, col=1
            )
        fig.add_trace(
            go.Scatter(x=t_eval_np, y=Vcorr_pred, 
                      name='Vcorr (predicted)', 
                      line=dict(width=2)),
            row=1, col=1
        )
        
        # Add current to top plot (right y-axis)
        if 'SOC' in data_dict:
            current = np.array(data_dict['SOC'])  # De-normalize current
            fig.add_trace(
                go.Scatter(x=t_eval_np, y=current, 
                          name='SOC', 
                          line=dict(width=1.5, color='gray', dash='dash'), 
                          opacity=0.5),
                row=1, col=1,
                secondary_y=True  # Right y-axis
            )
        
        # Vtotal comparison (bottom plot)
        if Vtotal_meas is not None and Vtotal_pred is not None:
            fig.add_trace(
                go.Scatter(x=t_eval_np, y=Vtotal_meas, 
                          name='Vref (measured)', 
                          line=dict(width=2), 
                          opacity=0.7),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=t_eval_np, y=Vtotal_pred, 
                          name='Vtotal (predicted)', 
                          line=dict(width=2)),
                row=2, col=1
            )
            
            # Add current to bottom plot (right y-axis)
            if 'SOC' in data_dict:
                current = np.array(data_dict['SOC'])  # De-normalize current
                fig.add_trace(
                    go.Scatter(x=t_eval_np, y=current, 
                              name='SOC', 
                              line=dict(width=1.5, color='gray', dash='dash'), 
                              opacity=0.5,
                              showlegend=False),  # Don't show in legend twice
                    row=2, col=1,
                    secondary_y=True  # Right y-axis
                )
        else:
            fig.add_trace(
                go.Scatter(x=t_eval_np, y=Vtotal_pred, 
                          name='Vtotal (predicted)', 
                          line=dict(width=2)),
                row=2, col=1
            )
            fig.update_yaxes(title_text='Voltage (V)', row=2, col=1)
        
        # Update axes labels
        fig.update_xaxes(title_text='Time (s)', row=1, col=1)
        fig.update_xaxes(title_text='Time (s)', row=2, col=1)
        fig.update_yaxes(title_text='Vcorr (V)', row=1, col=1)
        
        # Update secondary y-axes labels
        fig.update_yaxes(title_text='Current (A)', row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Current (A)', row=2, col=1, secondary_y=True)
        
        # Update layout
        fig.update_layout(
            height=900, 
            width=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.show()
    
    return Vcorr_pred, Vcorr_target, Vtotal_pred, Vtotal_meas, t_eval.cpu().numpy()


def analyze_feature_importance(ode_wrapper, data_dict, device='cuda', verbose=True):
    """
    Gradient-based feature importance analysis
    
    각 feature (V_ref, ocv, Vcorr, SOC, I, T)에 대한 gradient magnitude를 계산하여
    상대적 중요도를 평가합니다.
    
    Args:
        ode_wrapper: 트레이닝된 BatteryODEWrapper 모델
        data_dict: {'time': [...], 'V_ref': [...], 'ocv': [...], 'SOC': [...], 'I': [...], 'T': [...], 
                   'Y_mean': float, 'Y_std': float, 'V_spme': [...]}
        device: device to use
        verbose: print results
    
    Returns:
        importance_dict: {'V_ref': float, 'ocv': float, 'Vcorr': float, 'SOC': float, 'I': float, 'T': float}
                        각 feature의 상대적 중요도 (합계 = 1.0)
    """
    ode_wrapper.eval()
    ode_wrapper.set_inputs(data_dict)
    
    # Prepare data
    t_eval = torch.tensor(np.array(data_dict['time']), dtype=torch.float32, device=device)
    
    # Get initial condition
    if 'Y' in data_dict:
        x0_norm = torch.tensor([[data_dict['Y'][0]]], dtype=torch.float32, device=device)
    else:
        raise ValueError("data_dict must contain 'Y' for initial condition")
    
    # Sample points for gradient computation (use every Nth point for efficiency)
    sample_step = max(1, len(t_eval) // 100)  # 최대 100개 포인트만 사용
    t_sample = t_eval[::sample_step]
    
    # Storage for gradient magnitudes
    grad_magnitudes = {
        'V_ref': [],
        'ocv': [],
        'Vcorr': [],
        'SOC': [],
        'I': [],
        'T': []
    }
    
    # Feature names matching the input order
    feature_names = ['V_ref', 'ocv', 'Vcorr', 'SOC', 'I', 'T']
    
    # Compute gradients at sampled time points
    for t_val in t_sample:
        t_val_scalar = t_val.item()
        
        # Get current state (interpolate from solution or use target)
        with torch.no_grad():
            # Quick forward pass to get current Vcorr
            V_ref_k = torch.tensor([[float(ode_wrapper.inputs_interp['V_ref'](t_val_scalar))]], 
                                dtype=torch.float32, device=device)
            ocv_k = torch.tensor([[float(ode_wrapper.inputs_interp['ocv'](t_val_scalar))]], 
                                dtype=torch.float32, device=device)
            SOC_k = torch.tensor([[float(ode_wrapper.inputs_interp['SOC'](t_val_scalar))]], 
                                dtype=torch.float32, device=device)
            I_k = torch.tensor([[float(ode_wrapper.inputs_interp['I'](t_val_scalar))]], 
                              dtype=torch.float32, device=device)
            T_k = torch.tensor([[float(ode_wrapper.inputs_interp['T'](t_val_scalar))]], 
                              dtype=torch.float32, device=device)
            
            # Get approximate Vcorr by solving ODE up to this point
            t_subset = t_eval[t_eval <= t_val]
            if len(t_subset) > 0:
                with torch.no_grad():
                    solution_subset = odeint(ode_wrapper, x0_norm, t_subset, method='euler')
                    Vcorr_k = solution_subset[-1, 0, 0:1]
            else:
                Vcorr_k = x0_norm[0, 0:1]
        
        # Create input tensor that requires gradient
        V_ref_k.requires_grad = True
        ocv_k.requires_grad = True
        Vcorr_k.requires_grad = True
        SOC_k.requires_grad = True
        I_k.requires_grad = True
        T_k.requires_grad = True
        
        # Concatenate features: [V_ref, ocv, Vcorr, SOC, I, T]
        nn_input = torch.cat([V_ref_k, ocv_k, Vcorr_k, SOC_k, I_k, T_k], dim=1)
        
        # Forward pass
        output = ode_wrapper.net(nn_input)
        
        # Compute gradient for each feature
        for i, feat_name in enumerate(feature_names):
            grad = torch.autograd.grad(
                outputs=output,
                inputs=nn_input,
                grad_outputs=torch.ones_like(output),
                retain_graph=True,
                create_graph=False
            )[0]
            
            # Get gradient magnitude for this feature
            grad_mag = torch.abs(grad[0, i]).item()
            grad_magnitudes[feat_name].append(grad_mag)
    
    # Compute average gradient magnitude for each feature
    avg_grad_mags = {
        feat: np.mean(grad_magnitudes[feat]) 
        for feat in feature_names
    }
    
    # Normalize to relative importance (sum = 1.0)
    total = sum(avg_grad_mags.values())
    if total > 0:
        importance_dict = {
            feat: avg_grad_mags[feat] / total 
            for feat in feature_names
        }
    else:
        importance_dict = {feat: 0.25 for feat in feature_names}  # Equal if all zero
    
    # Print results
    if verbose:
        print(f"\n{'='*50}")
        print(f"Feature Importance Analysis (Gradient-based)")
        print(f"{'='*50}")
        print(f"Sampled {len(t_sample)} points from {len(t_eval)} total points")
        print(f"\nRelative Importance:")
        for feat in feature_names:
            print(f"  {feat:12s}: {importance_dict[feat]*100:6.2f}% "
                  f"(avg |grad|: {avg_grad_mags[feat]:.4e})")
        print(f"{'='*50}\n")
    
    return importance_dict


# ======== Convert Data ========
def struct_to_dataframe(mat_struct, selected_keys=None, add_vcorr=True):
    """
    MATLAB struct를 pandas DataFrame으로 변환
    
    Args:
        mat_struct: MATLAB struct from loadmat
        selected_keys: list of str or None
                      - None: 모든 key 사용
                      - ['key1', 'key2']: 원하는 key만 사용
        add_vcorr: bool
                  - True: Vcorr = Vref - Vspme 자동 계산
                  - False: Vcorr 계산 안함
    
    Returns:
        df: pandas DataFrame
    """
    struct_data = mat_struct[0, 0]
    
    # ===== 1. 모든 key 프린트 =====
    all_keys = struct_data.dtype.names
    print("=" * 60)
    print("Available keys in MATLAB struct:")
    print("=" * 60)
    for i, key in enumerate(all_keys, 1):
        print(f"  {i:2d}. {key}")
    print("=" * 60)
    print(f"Total: {len(all_keys)} keys\n")
    
    # ===== 2. 사용할 key 결정 =====
    if selected_keys is None:
        # 모든 key 사용
        keys_to_use = list(all_keys)
        print("→ Using ALL keys\n")
    else:
        # 지정된 key만 사용
        keys_to_use = [k for k in selected_keys if k in all_keys]
        missing_keys = [k for k in selected_keys if k not in all_keys]
        
        print("Selected keys:")
        for i, key in enumerate(keys_to_use, 1):
            print(f"  {i}. {key} ✓")
        
        if missing_keys:
            print(f"\n⚠ Warning: Keys not found in struct:")
            for key in missing_keys:
                print(f"  - {key} ✗")
        print()
    
    # ===== 3. key → DataFrame =====
    data_dict = {}
    print("Extracting data:")
    print("-" * 60)
    
    for key in keys_to_use:
        try:
            value = struct_data[key]
            
            # MATLAB double array → numpy array로 변환
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            
            # Flatten to 1D
            flattened = value.flatten()
            data_dict[key] = flattened
            print(f"  ✓ {key:20s}: shape {str(value.shape):15s} → {len(flattened)} points")
            
        except Exception as e:
            print(f"  ✗ {key:20s}: {str(e)}")
    
    print("-" * 60)
    
    # ===== 길이 체크 =====
    lengths = {key: len(data) for key, data in data_dict.items()}
    unique_lengths = set(lengths.values())
    
    if len(unique_lengths) > 1:
        print("\n⚠ WARNING: Arrays have different lengths!")
        print("Lengths:")
        for key, length in lengths.items():
            print(f"  {key:20s}: {length}")
        print("\n✗ Cannot create DataFrame with different length arrays.\n")
        return pd.DataFrame()
    
    # ===== 4. DataFrame 생성 =====
    if data_dict:
        df = pd.DataFrame(data_dict)
        
        # ===== 5. Vcorr 계산 (optional) =====
        if add_vcorr and 'Vref' in df.columns and 'Vspme' in df.columns:
            df['Vcorr'] = df['Vref'] - df['Vspme']
            print(f"\n✓ Vcorr added: Vcorr = Vref - Vspme")
        
        print(f"\n✓ DataFrame created successfully!")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}\n")
        return df
    else:
        print("\n✗ No data extracted. DataFrame is empty.\n")
        return pd.DataFrame()




def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_battery_neural_ode_batch(data_list, num_epochs=100, lr=1e-3, device='cpu', verbose=True, training_batch_size=None, ode_wrapper=None, method='euler'):
    """
    Train battery Neural ODE on multiple profiles (batch version)
    
    Args:
        data_list: List of data_dicts, each containing:
                  {'time', 'V_ref', 'ocv', 'SOC', 'I', 'T', 'V_spme', 'Y', 'Y_mean', 'Y_std'}
        num_epochs: number of training epochs
        lr: learning rate  
        device: device to use
        verbose: print progress
        training_batch_size: Number of profiles to use per training step (None = use all)
        ode_wrapper: Optional pre-existing BatteryODEWrapper instance to continue training from
        
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
    print(f"Batch Training: {total_profiles} total profiles")
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
    
    # Create ODE wrapper or use provided one
    if ode_wrapper is None:
        ode_wrapper = BatteryODEWrapper(device)
        ode_wrapper = ode_wrapper.to(device)
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
    print("Neural Network Architecture")
    print("="*60)
    print(ode_wrapper.net)
    print("="*60 + "\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(ode_wrapper.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)
    
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
        print("Starting batch training...")
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
        best_model_path = f"best_model_batch_rmse{best_rmse * Y_std_avg * 1000:.2f}mV.pth"
        
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




def test_battery_neural_ode_batch(
    data_list: Sequence[dict],
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    method: str = "euler",
    batch_size: Optional[int] = None,
    verbose: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run batch inference with a pretrained Neural ODE checkpoint (same batching as training).

    Args:
        data_list: list of profile dicts (same format as train_battery_neural_ode_batch)
        checkpoint_path: path to checkpoint (e.g. "best_model_batch_rmse1.63mV.pth")
        device: torch device
        method: ODE solver method ("euler", "rk4", ...)
        batch_size: optional profile batch size; None = all at once
        verbose: print progress

    Returns:
        preds_denorm_list: list of predicted Vcorr (denormalized) per profile
        targets_denorm_list: list of ground-truth Vcorr (denormalized) per profile
        vtotal_pred_list: list of predicted Vtotal (V_spme + Vcorr) per profile
        vtotal_gt_list: list of measured Vtotal (V_meas) per profile
    """
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    training_info = checkpoint.get("training_info", {})

    if verbose:
        print(f"\n[TEST] Loading checkpoint: {checkpoint_path}")
        if training_info:
            print("  --- Checkpoint Training Info ---")
            for key in sorted(training_info.keys()):
                print(f"  {key}: {training_info[key]}")
        else:
            print("  (no training_info metadata found)")
        if "network_architecture" in checkpoint:
            print("  --- Network Architecture ---")
            print(checkpoint["network_architecture"])
        print("=" * 60)

    # Create wrapper and load weights
    ode_wrapper = BatteryODEWrapper(device=device).to(device)
    ode_wrapper.load_state_dict(state_dict)
    ode_wrapper.eval()

    total_profiles = len(data_list)
    batch_size = batch_size or total_profiles
    batch_size = min(batch_size, total_profiles)

    if verbose:
        print(f"[TEST] Total profiles to evaluate: {total_profiles}")

    results: List[Dict[str, Any]] = []
    rmse_vcorr_list: List[float] = []
    rmse_vtotal_list: List[float] = []
    processed_profiles = 0
    progress_step = max(1, math.ceil(total_profiles * 0.1))
    next_progress = progress_step

    # Normalize each profile time to start at zero (same as training)
    normalized_list = []
    for profile in data_list:
        profile_norm = profile.copy()
        t_arr = np.array(profile["time"])
        profile_norm["time"] = t_arr - t_arr[0]
        normalized_list.append(profile_norm)

    # Process in batches
    for start in range(0, total_profiles, batch_size):
        end = min(start + batch_size, total_profiles)
        batch_profiles = normalized_list[start:end]

        # Find max length and duration for this batch
        lengths = [len(p["time"]) for p in batch_profiles]
        durations = [p["time"][-1] for p in batch_profiles]
        max_len = max(lengths)
        max_duration = max(durations)
        t_common = np.linspace(0, max_duration, max_len)
        t_eval = torch.tensor(t_common, dtype=torch.float32, device=device)

        # Prepare inputs for batch solver (same as training)
        inputs_list = []
        targets_list = []
        v_spme_list = []
        v_meas_list = []
        y_mean_list = []
        y_std_list = []

        for prof in batch_profiles:
            inputs_list.append({
                "time": prof["time"],
                "V_ref": prof["V_ref"],
                "ocv": prof["ocv"],
                "SOC": prof["SOC"],
                "I": prof["I"],
                "T": prof["T"],
                "V_spme": prof["V_spme"],
            })
            targets_list.append(prof["Y"])
            v_spme_list.append(prof.get("V_spme"))
            v_meas_list.append(prof.get("V_meas"))
            y_mean_list.append(prof.get("Y_mean"))
            y_std_list.append(prof.get("Y_std"))

        ode_wrapper.set_inputs_batch(inputs_list, t_common)
        ode_wrapper.batch_indices = None  # use full batch

        # Initial conditions (normalized Y)
        targets_padded = []
        valid_lengths = []
        for target in targets_list:
            valid_lengths.append(len(target))
            if len(target) < max_len:
                padded = np.concatenate([target, np.full(max_len - len(target), target[-1])])
            else:
                padded = target[:max_len]
            targets_padded.append(padded)
        targets_batch = torch.tensor(np.array(targets_padded), dtype=torch.float32, device=device)
        x0 = targets_batch[:, 0:1]

        with torch.no_grad():
            ode_wrapper.step_count = 0
            solution = odeint(ode_wrapper, x0, t_eval, method=method)  # [T, batch, 1]
            pred_norm = solution[:, :, 0].T  # [batch, T]

        # Unpad / denormalize per profile
        for idx, prof in enumerate(batch_profiles):
            global_idx = start + idx
            valid_len = valid_lengths[idx]
            y_mean = y_mean_list[idx]
            y_std = y_std_list[idx]
            pred_norm_profile = pred_norm[idx, :valid_len].cpu().numpy()
            targ_norm_profile = np.array(targets_list[idx][:valid_len])

            if y_mean is None or y_std is None:
                pred_denorm = pred_norm_profile
                targ_denorm = targ_norm_profile
            else:
                pred_denorm = pred_norm_profile * y_std + y_mean
                targ_denorm = targ_norm_profile * y_std + y_mean

            if v_spme_list[idx] is not None:
                v_spme = np.array(v_spme_list[idx][:valid_len])
                vtotal_pred = v_spme + pred_denorm
            else:
                v_spme = None
                vtotal_pred = None

            if v_meas_list[idx] is not None:
                v_meas = np.array(v_meas_list[idx][:valid_len])
                vtotal_gt = v_meas
            else:
                v_meas = None
                vtotal_gt = None

            rmse_vcorr = float(np.sqrt(np.mean((pred_denorm - targ_denorm) ** 2)) * 1000)
            rmse_vtotal = (
                float(np.sqrt(np.mean((vtotal_pred - vtotal_gt) ** 2)) * 1000)
                if vtotal_pred is not None and vtotal_gt is not None
                else None
            )

            rmse_vcorr_list.append(rmse_vcorr)
            if rmse_vtotal is not None:
                rmse_vtotal_list.append(rmse_vtotal)

            results.append(
                {
                    "profile_index": global_idx,
                    "time": prof["time"][:valid_len],
                    "Vcorr_pred": pred_denorm,
                    "Vcorr_target": targ_denorm,
                    "Vtotal_pred": vtotal_pred,
                    "Vtotal_target": vtotal_gt,
                    "V_meas": v_meas,
                    "V_spme": v_spme,
                    "SOC": np.array(prof.get("SOC", [np.nan] * valid_len))[:valid_len],
                    "rmse_vcorr_mV": rmse_vcorr,
                    "rmse_vtotal_mV": rmse_vtotal,
                    "num_points": valid_len,
                }
            )
            processed_profiles += 1

            if verbose and processed_profiles >= next_progress:
                pct = min(100, int(round(processed_profiles / total_profiles * 100)))
                print(f"[TEST] Progress: {processed_profiles}/{total_profiles} profiles ({pct}%)")
                next_progress += progress_step

        if verbose and results:
            last_rmse = results[-1]["rmse_vcorr_mV"]
            print(f"[TEST] Profiles {start}-{end - 1}: last RMSE = {last_rmse:.2f} mV")

    summary: Dict[str, Any] = {
        "num_profiles": total_profiles,
        "avg_rmse_vcorr_mV": float(np.mean(rmse_vcorr_list)) if rmse_vcorr_list else float("nan"),
        "median_rmse_vcorr_mV": float(np.median(rmse_vcorr_list)) if rmse_vcorr_list else float("nan"),
        "avg_rmse_vtotal_mV": float(np.mean(rmse_vtotal_list)) if rmse_vtotal_list else None,
        "median_rmse_vtotal_mV": float(np.median(rmse_vtotal_list)) if rmse_vtotal_list else None,
        "checkpoint_path": str(checkpoint_path),
        "solver": method,
        "batch_size": batch_size,
    }

    if verbose:
        print(f"[TEST] Completed {total_profiles} profiles (batch_size={batch_size}, solver={method})")
        print(
            f"[TEST] RMSE Vcorr -> avg: {summary['avg_rmse_vcorr_mV']:.2f} mV, "
            f"median: {summary['median_rmse_vcorr_mV']:.2f} mV"
        )
        if summary["avg_rmse_vtotal_mV"] is not None:
            print(
                f"[TEST] RMSE Vtotal -> avg: {summary['avg_rmse_vtotal_mV']:.2f} mV, "
                f"median: {summary['median_rmse_vtotal_mV']:.2f} mV"
            )

    if save_path is not None:
        save_payload = {
            "results": results,
            "summary": summary,
            "training_info": training_info,
        }
        torch.save(save_payload, save_path)
        if verbose:
            print(f"[TEST] Saved detailed results to: {save_path}")

    return results, summary