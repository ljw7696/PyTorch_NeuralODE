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



class BatteryODEWrapper(nn.Module):
    """
    Neural ODE wrapper for battery voltage correction - mirrors MATLAB ode_wrapper_poly
    
    This replaces the physics-based solid_dynamics_poly and electrolyte_dynamics_poly
    with a neural network that learns dVcorr/dt from battery states.
    
    Inputs:
        t       : current time [s] 
        x       : state vector [1x1] = [Vcorr]
        inputs  : dict with time-varying inputs (I(t), T(t), csn_bulk(t))
        params  : neural network parameters (self.net)
    Output:
        dxdt    : time derivative of state vector [1x1] = [dVcorr/dt]
    """
    
    def __init__(self, device='cpu'):
        super(BatteryODEWrapper, self).__init__()
        
        # Neural network: input [csn_bulk(t), I(t), T(t), Vcorr(t)] -> output dVcorr/dt
        # 4 inputs: [Vcorr_k, csn_bulk, I, T]
        # 7 layers: original working version
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1.0)
                # nn.init.constant_(m.bias, val=0)
                nn.init.orthogonal_(m.weight, gain=0.5)  # Orthogonal!
                nn.init.constant_(m.bias, 0)
        
        self.device = device
        
        # Store interpolation functions for time-varying inputs
        self.inputs_interp = None
        
        # Initialize step_count for tracking
        self.step_count = 0
        
        # # Initialize prev_Vcorr for state tracking
        # self.prev_Vcorr = None
        
    def set_inputs(self, inputs_dict):
        """
        Set up interpolation functions for time-varying inputs
        Similar to MATLAB inputs struct with I(t), T(t), etc.
        
        Args:
            inputs_dict: {'time': [...], 'I': [...], 'T': [...], 'csn_bulk': [...]}
        """
        time_data = inputs_dict['time']
        
        self.inputs_interp = {
            'I': interp1d(time_data, inputs_dict['I'], kind='linear', fill_value='extrapolate'),
            'T': interp1d(time_data, inputs_dict['T'], kind='linear', fill_value='extrapolate'), 
            'csn_bulk': interp1d(time_data, inputs_dict['csn_bulk'], kind='linear', fill_value='extrapolate'),
            'V_spme': interp1d(time_data, inputs_dict['V_spme'], kind='linear', fill_value='extrapolate'),
        }
        
        # # Reset prev_Vcorr when inputs change
        # self.prev_Vcorr = None
    
    def forward(self, t, x):
        """
        ODE function called by odeint - mirrors MATLAB ode_wrapper_poly
        
        Neural ODE: dVcorr/dt = f(Vcorr, csn_bulk, I, T)
        where f() is the neural network
        
        Args:
            t: current time (scalar)
            x: state vector [Vcorr] (1x1 tensor)
            
        Returns:
            dxdt: derivative [dVcorr/dt] (1x1 tensor)
        """
        # Convert time to scalar if needed
        if isinstance(t, torch.Tensor):
            t_val = t.item()
        else:
            t_val = float(t)

        # Extract current state: Vcorr(k)
        Vcorr_k = x[0, 0].item() if isinstance(x, torch.Tensor) else float(x)


        # Evaluate time-varying inputs at time t
        if self.inputs_interp is None:
            raise ValueError("Must call set_inputs() before solving ODE")
            
        csn_k = float(self.inputs_interp['csn_bulk'](t_val))  # csn_bulk(k)
        I_k = float(self.inputs_interp['I'](t_val))           # I(k)
        T_k = float(self.inputs_interp['T'](t_val))           # T(k)
        Vspme_k = float(self.inputs_interp['V_spme'](t_val))   # Vspme(k)
        
        # Neural Network Input: X = [Vcorr(k), csn_bulk(k), I(k), T(k)]
        nn_input = torch.tensor([[Vcorr_k, csn_k, I_k, T_k]], dtype=torch.float32, device=self.device)
        
        # Neural Network Output: dVcorr/dt(k)
        dVcorr_dt_k = self.net(nn_input)
        
        # 부호 반전 (처음에 방향 반대로 나와서 적용)
        dVcorr_dt_k = dVcorr_dt_k

        self.step_count += 1
        
        # Return derivative for ODE solver
        return dVcorr_dt_k




def train_battery_neural_ode(data_dict, num_epochs=100, lr=1e-3, device='cpu', verbose=True, pretrained_model_path=None):
    """
    Train battery Neural ODE - mirrors MATLAB ode15s usage
    
    Args:
        data_dict: {'time', 'csn_bulk', 'I', 'T', 'V_spme', 'V_meas'}
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
        data_dict: {'time': [...], 'csn_bulk': [...], 'I': [...], 'T': [...], 
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
        V_spme_norm = np.array(data_dict['V_spme'])
        
        # De-normalize V_spme
        if 'V_spme_mean' in data_dict and 'V_spme_std' in data_dict:
            V_spme_mean = data_dict['V_spme_mean']
            V_spme_std = data_dict['V_spme_std']
            V_spme = V_spme_norm * V_spme_std + V_spme_mean
        else:
            V_spme = V_spme_norm
            print("Warning: No V_spme_mean/V_spme_std found, using normalized V_spme")
        
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
    
    각 feature (Vcorr, csn_bulk, I, T)에 대한 gradient magnitude를 계산하여
    상대적 중요도를 평가합니다.
    
    Args:
        ode_wrapper: 트레이닝된 BatteryODEWrapper 모델
        data_dict: {'time': [...], 'csn_bulk': [...], 'I': [...], 'T': [...], 
                   'Y_mean': float, 'Y_std': float, 'V_spme': [...]}
        device: device to use
        verbose: print results
    
    Returns:
        importance_dict: {'Vcorr': float, 'csn_bulk': float, 'I': float, 'T': float}
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
        'Vcorr': [],
        'csn_bulk': [],
        'I': [],
        'T': []
    }
    
    # Feature names matching the input order
    feature_names = ['Vcorr', 'csn_bulk', 'I', 'T']
    
    # Compute gradients at sampled time points
    for t_val in t_sample:
        t_val_scalar = t_val.item()
        
        # Get current state (interpolate from solution or use target)
        with torch.no_grad():
            # Quick forward pass to get current Vcorr
            csn_k = torch.tensor([[float(ode_wrapper.inputs_interp['csn_bulk'](t_val_scalar))]], 
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
        Vcorr_k.requires_grad = True
        csn_k.requires_grad = True
        I_k.requires_grad = True
        T_k.requires_grad = True
        
        # Concatenate features
        nn_input = torch.cat([Vcorr_k, csn_k, I_k, T_k], dim=1)
        
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


# Example usage (similar to MATLAB):
"""
# MATLAB equivalent:
[t_sol, x_sol] = ode15s(@(t, x) ode_wrapper_poly(t, x, inputs, battery_params), inputs.t, x0, opts);

# Python equivalent:
ode_wrapper = BatteryODEWrapper(device)
ode_wrapper.set_inputs(inputs_dict)
solution = odeint(ode_wrapper, x0, t_eval)
"""



def smooth_Vcorr(Y, window_size=21):
    """
    Smooth Vcorr using Savitzky-Golay filter
    
    Args:
        Y: array-like, Vcorr data to smooth
        window_size: int, window length (must be odd, default=21)
    
    Returns:
        Y_smooth: smoothed Vcorr data
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"Warning: window_size must be odd. Using {window_size} instead.")
    
    # Smooth
    Y_smooth = savgol_filter(
        Y,
        window_length=window_size,
        polyorder=3,
        mode='interp'
    )
    
    return Y_smooth



def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5