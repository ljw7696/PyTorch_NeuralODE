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
import re


# ======== Convert Data ========
def struct2df(mat_struct, selected_keys=None, add_vcorr=True):
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


def downsample_df(df_list, downsample_rate):
    """
    Downsample a list of DataFrames by a given rate.
    
    Args:
        df_list: list of pandas DataFrames to downsample (or single DataFrame)
        downsample_rate: int, rate to downsample (e.g., 3 means reduce to 1/3 of original length)
    
    Returns:
        list of downsampled DataFrames (or single DataFrame if input was single)
    """
    if downsample_rate <= 0:
        raise ValueError("downsample_rate must be positive")
    
    # Handle single DataFrame input
    if isinstance(df_list, pd.DataFrame):
        df = df_list
        if downsample_rate == 1:
            return df
        
        # Calculate target length: original / downsample_rate
        target_length = len(df) // downsample_rate
        
        if target_length == 0:
            return df.iloc[:1].reset_index(drop=True)  # Return at least 1 point
        
        # Uniformly downsample DataFrame to target_length points
        idxs = np.round(np.linspace(0, len(df) - 1, target_length)).astype(int)
        df = df.iloc[idxs].reset_index(drop=True)
        
        return df
    
    # Handle list of DataFrames
    if not isinstance(df_list, list):
        raise ValueError("Input must be a list of DataFrames or a single DataFrame")
    
    if len(df_list) == 0:
        return []
    
    result_list = []
    
    for df in df_list:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("All items in list must be DataFrames")
        
        if downsample_rate == 1:
            result_list.append(df)
            continue
        
        # Calculate target length: original / downsample_rate
        target_length = len(df) // downsample_rate
        
        if target_length == 0:
            result_list.append(df.iloc[:1].reset_index(drop=True))  # Return at least 1 point
            continue
        
        # Uniformly downsample DataFrame to target_length points
        idxs = np.round(np.linspace(0, len(df) - 1, target_length)).astype(int)
        df_ds = df.iloc[idxs].reset_index(drop=True)
        result_list.append(df_ds)
    
    return result_list


def smooth_Vcorr(df_list, window_size=21):
    """
    Smooth Vcorr for a list of DataFrames using Savitzky-Golay filter.
    
    Args:
        df_list: list of pandas DataFrames, each must contain 'Vref' and 'Vspme' columns
        window_size: int, window length (must be odd, default=21)
    
    Returns:
        list of DataFrames with smoothed Vcorr data
    """
    if not isinstance(df_list, list):
        raise ValueError("Input must be a list of DataFrames")
    
    if len(df_list) == 0:
        return []
    
    print("="*50)
    print("Starting 'smooth_Vcorr()'")
    print(f"Original length of df : {len(df_list)}")
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    result_list = []
    
    for idx, df in enumerate(df_list):
        original_length = len(df)
        
        # Calculate Vcorr
        Vcorr = df['Vref'].values.astype(np.float32) - df['Vspme'].values.astype(np.float32)
        
        # Smooth Vcorr if data is long enough
        df_result = df.copy(deep=True)
        if len(Vcorr) < window_size:
            print(f"Warning: Data length is less than window_size, skipping smoothing. window_size: {window_size}, data length: {len(Vcorr)}")
            df_result['Vcorr'] = Vcorr
        else:
            Vcorr_smooth = savgol_filter(
                Vcorr,
                window_length=window_size,
                polyorder=3,
                mode='interp'
            )
            df_result['Vcorr'] = Vcorr_smooth
        
        final_length = len(df_result)
        print(f"- df{idx+1}: {original_length} -> {final_length} (smoothing done)")
        
        result_list.append(df_result)
    
    
    print("smooth_Vcorr() completed")
    print("="*50)
    
    return result_list


def df2dict(df_driving_list, df_rest_list=None):
    """
    Convert a DataFrame or list of DataFrames to data dicts with keys specified by `keys`.
    Normalization/scaling is left as a placeholder for user to fill (see main.ipynb 156-172).

    Args:
        df_driving_list: list of DataFrames containing driving/any battery data
        df_rest_list: Optional. list of DataFrames containing rest battery data (None = skip rest processing)

    Returns:
        If df_rest_list is None: returns (dict_list_driving,)
        Otherwise: returns (dict_list_driving, dict_list_rest)

    Note:
        - If a key is missing in a DataFrame, value will be None. 
        - You should handle normalization/scaling in user code.
    """
    # If only one argument provided, treat as driving_list only
    if df_rest_list is None:
        print("="*50)
        print("Starting 'df2dict()'")
        print(f"Total number of df in the list: {len(df_driving_list)}")
        
        # Process as driving only
        csn_bulk_norm = 30555
        current_norm = 0.1
        temperature_norm = 10
        temperature_mean = 298.15
        Y_mean = -0.015
        Y_std = 0.004
        ocv_mean = 3.28 # 96 - 12 % SOC only
        ocv_std = 0.04 # 96 - 12 % SOC only 
        Vref_mean = 3.28 - 0.015 #
        Vref_std = 0.04 # 

                
        dict_list_driving = []
        
        for idx, df_cur in enumerate(df_driving_list):
            Vcorr = df_cur['Vcorr'].values.astype(np.float32)
            
            data_dict = {}
            # Inputs
            data_dict['SOC'] = df_cur['soc_n'].values.astype(np.float32)  # SOC: 0~1, already normalized
            data_dict['I'] = df_cur['current'].values.astype(np.float32) / current_norm
            data_dict['T'] = (df_cur['temperature'].values.astype(np.float32) - temperature_mean) / temperature_norm
            data_dict['V_ref'] = (df_cur['Vref'].values.astype(np.float32) - Vref_mean) / Vref_std
            data_dict['ocv'] = (df_cur['ocp'].values.astype(np.float32) - ocv_mean) / ocv_std
            data_dict['V_spme_norm'] = (df_cur['Vspme'].values.astype(np.float32) - 3.28) / Vref_std


            # Outputs
            data_dict['Y'] = (Vcorr - Y_mean) / Y_std
            data_dict['Y_mean'] = Y_mean
            data_dict['Y_std'] = Y_std

            # Others
            data_dict['time'] = df_cur['time'].values.astype(np.float32)
            # data_dict['csn_bulk'] = df_cur['c_s_n_bulk'].values.astype(np.float32) / csn_bulk_norm  # Keep for reference
            data_dict['V_meas'] = df_cur['Vref'].values.astype(np.float32)
            data_dict['V_spme'] = df_cur['Vspme'].values.astype(np.float32)
            data_dict['Vcorr'] = Vcorr
            dict_list_driving.append(data_dict)


            # Normalization Information
            data_dict['csn_bulk_norm'] = csn_bulk_norm
            data_dict['current_norm'] = current_norm
            data_dict['temperature_norm'] = temperature_norm
            data_dict['temperature_mean'] = temperature_mean
            data_dict['Y_mean'] = Y_mean
            data_dict['Y_std'] = Y_std
            data_dict['ocv_mean'] = ocv_mean
            data_dict['ocv_std'] = ocv_std
            data_dict['Vref_mean'] = Vref_mean
            data_dict['Vref_std'] = Vref_std
        
        print(f"Number of converted df: {len(dict_list_driving)}")
        print("df2dict() completed")
        print("="*50)
        return dict_list_driving
    
    # Original two-argument processing
    print("="*50)
    print("Starting 'df2dict()'")
    total_df_count = len(df_driving_list) + (len(df_rest_list) if df_rest_list is not None else 0)
    print(f"Total number of df in the list : {total_df_count}")
    
    csn_bulk_norm = 30555 # csn_max
    current_norm = 0.1 # average current
    temperature_norm = 10 # temperature range 15, 25, 35, 45C -> 10C gap
    temperature_mean = 298.15 # 25C in K
    Y_mean = -0.015
    Y_std = 0.004
    ocv_mean = 3.28 # 96 - 12 % SOC only
    ocv_std = 0.04 # 96 - 12 % SOC only 
    Vref_mean = 3.28 - 0.015 #
    Vref_std = 0.04 # 
    
    # Loop through both df_driving_list and df_rest_list to process them separately
    
    # Make sure both inputs are lists (not single DataFrame)
    # Each list element is a DataFrame
    dict_list_driving = []
    dict_list_rest = []

    # Process driving list
    for idx, df_cur in enumerate(df_driving_list):
        Vcorr = df_cur['Vcorr'].values.astype(np.float32)
        # Driving 데이터는 smoothing X, downsample X (원형 유지 가정)
        
        data_dict = {}
        # Inputs
        data_dict['SOC'] = df_cur['soc_n'].values.astype(np.float32)  # SOC: 0~1, already normalized
        data_dict['I'] = df_cur['current'].values.astype(np.float32) / current_norm
        data_dict['T'] = (df_cur['temperature'].values.astype(np.float32) - temperature_mean) / temperature_norm
        data_dict['V_ref'] = (df_cur['Vref'].values.astype(np.float32) - Vref_mean) / Vref_std
        data_dict['ocv'] = (df_cur['ocp'].values.astype(np.float32) - ocv_mean) / ocv_std
        data_dict['V_spme_norm'] = (df_cur['Vspme'].values.astype(np.float32) - 3.28) / Vref_std


        # Outputs
        data_dict['Y'] = (Vcorr - Y_mean) / Y_std
        data_dict['Y_mean'] = Y_mean
        data_dict['Y_std'] = Y_std

        # Others
        data_dict['time'] = df_cur['time'].values.astype(np.float32)
        # data_dict['csn_bulk'] = df_cur['c_s_n_bulk'].values.astype(np.float32) / csn_bulk_norm  # Keep for reference
        data_dict['V_meas'] = df_cur['Vref'].values.astype(np.float32)
        data_dict['V_spme'] = df_cur['Vspme'].values.astype(np.float32)
        dict_list_driving.append(data_dict)

        # Normalization Information
        data_dict['csn_bulk_norm'] = csn_bulk_norm
        data_dict['current_norm'] = current_norm
        data_dict['temperature_norm'] = temperature_norm
        data_dict['temperature_mean'] = temperature_mean
        data_dict['Y_mean'] = Y_mean
        data_dict['Y_std'] = Y_std
        data_dict['ocv_mean'] = ocv_mean
        data_dict['ocv_std'] = ocv_std
        data_dict['Vref_mean'] = Vref_mean
        data_dict['Vref_std'] = Vref_std

    # Process rest list
    if len(df_rest_list) == 0:
        # 빈 리스트면 스킵 (첫 번째 리스트만 처리)
        pass
    elif len(df_rest_list) > len(df_driving_list):
        # Rest list가 더 길면 경고하고 처음 len(df_driving_list)개만 처리
        df_rest_list = df_rest_list[:len(df_driving_list)]
    
    for idx, df_cur in enumerate(df_rest_list):
        # Rest case: no downsampling (downsampling should be done separately before calling df2dict)
        Vcorr = df_cur['Vcorr'].values.astype(np.float32)

        data_dict = {}
        # Inputs
        data_dict['SOC'] = df_cur['soc_n'].values.astype(np.float32)  # SOC: 0~1, already normalized
        data_dict['I'] = df_cur['current'].values.astype(np.float32) / current_norm
        data_dict['T'] = (df_cur['temperature'].values.astype(np.float32) - temperature_mean) / temperature_norm
        data_dict['V_ref'] = (df_cur['Vref'].values.astype(np.float32) - Vref_mean) / Vref_std
        data_dict['ocv'] = (df_cur['ocp'].values.astype(np.float32) - ocv_mean) / ocv_std
        data_dict['V_spme_norm'] = (df_cur['Vspme'].values.astype(np.float32) - 3.28) / Vref_std

        # Outputs
        data_dict['Y'] = (Vcorr - Y_mean) / Y_std
        data_dict['Y_mean'] = Y_mean
        data_dict['Y_std'] = Y_std

        # Others
        data_dict['time'] = df_cur['time'].values.astype(np.float32)
        data_dict['V_meas'] = df_cur['Vref'].values.astype(np.float32)
        data_dict['V_spme'] = df_cur['Vspme'].values.astype(np.float32)
        dict_list_rest.append(data_dict)

        # Normalization Information
        data_dict['csn_bulk_norm'] = csn_bulk_norm
        data_dict['current_norm'] = current_norm
        data_dict['temperature_norm'] = temperature_norm
        data_dict['temperature_mean'] = temperature_mean
        data_dict['Y_mean'] = Y_mean
        data_dict['Y_std'] = Y_std
        data_dict['ocv_mean'] = ocv_mean
        data_dict['ocv_std'] = ocv_std
        data_dict['Vref_mean'] = Vref_mean
        data_dict['Vref_std'] = Vref_std

    total_converted = len(dict_list_driving) + len(dict_list_rest)
    print(f"Number of converted df : {total_converted}")
    print("df2dict() completed")
    print("="*50)

    # 최종 반환값: (운전자용 dict list, 휴식용 dict list) 튜플로 반환
    return dict_list_driving, dict_list_rest


def downsample_evenly(dict_list, factor):
    """
    Downsample a list of profile dictionaries evenly by the given factor.

    Args:
        dict_list: list of dicts. Each dict contains time-series arrays with the
                   same primary length (e.g. 'time', 'V_meas', 'SOC', ...).
        factor: int >= 1. A factor of 2 keeps ~50% of the points, 3 keeps ~33%, etc.

    Returns:
        list of dicts with arrays downsampled along the first axis.
    """
    if not isinstance(dict_list, list):
        raise ValueError("dict_list must be a list of dictionaries.")

    if factor < 1:
        raise ValueError("factor must be >= 1.")

    if factor == 1:
        # return shallow copies to avoid accidental in-place edits
        return [copy.deepcopy(profile) for profile in dict_list]

    downsampled_list = []

    for profile in dict_list:
        if not isinstance(profile, dict):
            raise ValueError("Each item in dict_list must be a dictionary.")

        # Determine original length using 'time' if available, otherwise the first array-like entry.
        length = None
        time_values = profile.get("time", None)

        if isinstance(time_values, np.ndarray) and time_values.ndim > 0:
            length = len(time_values)
        elif torch.is_tensor(time_values) and time_values.ndim > 0:
            length = time_values.shape[0]
        elif isinstance(time_values, (list, tuple)):
            length = len(time_values)

        if length is None:
            # fallback: find first array-like value
            for value in profile.values():
                if isinstance(value, np.ndarray) and value.ndim > 0:
                    length = len(value)
                    break
                if torch.is_tensor(value) and value.ndim > 0:
                    length = value.shape[0]
                    break
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    length = len(value)
                    break

        if length is None or length == 0:
            # nothing to downsample
            downsampled_list.append(copy.deepcopy(profile))
            continue

        target_len = max(1, int(np.ceil(length / factor)))
        if target_len >= length:
            downsampled_list.append(copy.deepcopy(profile))
            continue

        idx = np.linspace(0, length - 1, num=target_len, endpoint=True, dtype=np.int64)

        def _downsample_value(val):
            if isinstance(val, np.ndarray):
                if val.ndim == 0:
                    return val
                valid_idx = idx[idx < val.shape[0]]
                return val[valid_idx]
            if torch.is_tensor(val):
                if val.ndim == 0:
                    return val
                valid_idx = idx[idx < val.shape[0]]
                if len(valid_idx) == 0:
                    return val[:1]
                index_tensor = torch.as_tensor(valid_idx, device=val.device, dtype=torch.long)
                return val.index_select(0, index_tensor)
            if isinstance(val, list):
                return [val[i] for i in idx if i < len(val)]
            if isinstance(val, tuple):
                return tuple(val[i] for i in idx if i < len(val))
            # leave scalars / unknown types untouched
            return val

        new_profile = {}
        for key, value in profile.items():
            new_profile[key] = _downsample_value(value)

        downsampled_list.append(new_profile)

    return downsampled_list


def zero_order_hold_dict_list(dict_list, hold_factor, keys_to_hold=None):
    """
    Apply zero-order hold to selected entries in a list of profile dictionaries.

    Args:
        dict_list: list of dicts containing profile data.
        hold_factor: number of samples per hold segment (>=1).
        keys_to_hold: iterable of keys to process. If None, all array-like entries
                      are held.

    Returns:
        List of dicts with zero-order held values.
    """
    if not isinstance(dict_list, list):
        raise ValueError("dict_list must be a list of dictionaries.")
    if hold_factor < 1:
        raise ValueError("hold_factor must be >= 1.")

    held_list = []
    target_keys = set(keys_to_hold) if keys_to_hold is not None else None

    for profile in dict_list:
        if not isinstance(profile, dict):
            raise ValueError("Each item in dict_list must be a dictionary.")

        new_profile = copy.deepcopy(profile)

        for key, value in profile.items():
            if target_keys is not None and key not in target_keys:
                continue

            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.size == 0 or hold_factor == 1:
                    continue
                held = value.copy()
                for start in range(0, len(held), hold_factor):
                    held[start:start + hold_factor] = held[start]
                new_profile[key] = held

            elif torch.is_tensor(value):
                if value.ndim == 0 or value.numel() == 0 or hold_factor == 1:
                    continue
                held = value.clone()
                length = held.shape[0]
                for start in range(0, length, hold_factor):
                    held[start:start + hold_factor] = held[start]
                new_profile[key] = held

            elif isinstance(value, list):
                if len(value) == 0 or hold_factor == 1:
                    continue
                held = value[:]
                for start in range(0, len(held), hold_factor):
                    segment_value = held[start]
                    end = min(start + hold_factor, len(held))
                    for idx in range(start, end):
                        held[idx] = segment_value
                new_profile[key] = held

            elif isinstance(value, tuple):
                if len(value) == 0 or hold_factor == 1:
                    continue
                held_list_tuple = list(value)
                for start in range(0, len(held_list_tuple), hold_factor):
                    segment_value = held_list_tuple[start]
                    end = min(start + hold_factor, len(held_list_tuple))
                    for idx in range(start, end):
                        held_list_tuple[idx] = segment_value
                new_profile[key] = tuple(held_list_tuple)

        held_list.append(new_profile)

    return held_list


def compute_grad_norm(model):
    """Compute gradient norm of model parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def remove_duplicates(df_list, verbose=True):
    """
    Remove duplicate time points from DataFrame(s)
    
    Args:
        df_list: list of pandas DataFrames, each must contain 'time' column
        verbose: bool, print progress if True
    
    Returns:
        list of cleaned DataFrames with duplicates removed
    
    Each DataFrame is cleaned by removing duplicate time points.
    """
    if not isinstance(df_list, list):
        raise ValueError("Input must be a list of DataFrames")
    
    if len(df_list) == 0:
        return []
    
    if verbose:
        print("="*50)
        print("Starting 'remove_duplicates()'")
        print(f"Original length of df : {len(df_list)}")
    
    cleaned_list = []
    
    for idx, df in enumerate(df_list):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Item {idx} is not a DataFrame")
        
        if 'time' not in df.columns:
            raise ValueError(f"DataFrame {idx} missing 'time' column")
        
        time_float = df['time'].astype(np.float32).values
        original_length = len(time_float)
        
        # Find problem indices (duplicates or decreasing)
        diff = np.diff(time_float)
        problem_idx = np.where(diff <= 0)[0]
        
        # Determine indices to remove
        indices_to_remove = []
        for p_idx in problem_idx:
            t0 = time_float[p_idx]
            t1 = time_float[p_idx + 1]
            diff_val = t1 - t0
            
            if abs(diff_val) < 1e-10:  # duplicate
                indices_to_remove.append(p_idx + 1)  # Remove second index
            else:  # decreasing
                indices_to_remove.append(p_idx + 1)  # Remove second index
        
        # Remove duplicates
        indices_to_remove = sorted(set(indices_to_remove))  # Remove duplicates and sort
        
        # Drop rows by index
        cleaned_df = df.drop(index=df.index[indices_to_remove]).reset_index(drop=True)
        
        time_clean = cleaned_df['time'].astype(np.float32).values
        final_length = len(time_clean)
        removed_count = original_length - final_length
        
        if verbose:
            print(f"- df{idx+1}: {original_length} -> {final_length} (removed {removed_count} points)")
        
        cleaned_list.append(cleaned_df)
    
    if verbose:
        print("remove_duplicates() completed")
        print("="*50)
    
    return cleaned_list


def split_df(list_of_df, window_minutes=30, time_col='time', random_seed=None):
    """
    각 DataFrame에서 순차적으로 30분(1800초) 구간씩 겹치지 않게 쪼개서 반환.

    Args:
        list_of_df: list of pd.DataFrame
        window_minutes: 쪼갤 구간 (분)
        time_col: 시간 컬럼명 (기본값 'time')
        random_seed: int or None (현재 사용 안함, 나중을 위해 유지)

    Returns:
        return_list: list of pd.DataFrame (각각 쪼개진 30분 구간)
    
    Example:
        >>> driving_list = [df_driving1, df_driving2]
        >>> rest_list = [df_rest1, df_rest2]
        >>> all_chunks = split_df(driving_list + rest_list, window_minutes=30)
        >>> # [df_driving1_chunk1, df_driving1_chunk2, df_driving2_chunk1, df_rest1_chunk1, ...]
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    print("="*50)
    print("Starting 'split_df()'")
    print(f"Number of df in the list : {len(list_of_df)}")
    
    # Calculate total time across all DataFrames
    total_time = 0
    for df in list_of_df:
        times = df[time_col].values
        if len(times) > 0:
            total_time += times[-1] - times[0]
    
    print(f"Total time in the list: {total_time:.2f} sec")
    print(f"window_minutes: {window_minutes} min")

    window_seconds = window_minutes * 60
    return_list = []
    
    for df in list_of_df:
        times = df[time_col].values
        df_len = len(df)
        
        # window가 데이터 길이보다 크면 입력 그대로 리턴
        if df_len > 0:
            total_time_span = times[-1] - times[0]
            if window_seconds > total_time_span:
                return_list.append(df.copy())
            continue
        
        st = 0  # 시작 인덱스

        # 시간 직접 체크 (순차적 방식)
        while st < df_len:
            curr_time = times[st]
            
            # 윈도우 종료 시간 계산
            end_time = curr_time + window_seconds

            # end_time 안 넘는 인덱스까지 포함 (마지막 점 포함)
            # st 이후의 인덱스만 확인
            time_diffs = times[st:] - curr_time
            valid_mask = time_diffs <= window_seconds
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # 마지막 조각 추가 (window보다 작은 남은 데이터)
                if st < df_len:
                    return_list.append(df.iloc[st:].reset_index(drop=True))
                break
            
            # valid_indices는 times[st:]의 상대 인덱스이므로, 원래 인덱스로 변환
            end_idx = st + valid_indices[-1]

            return_list.append(df.iloc[st:end_idx+1].reset_index(drop=True))

            # 다음 시작 지점은 end_idx + 1 (겹치지 않게)
            st = end_idx + 1

    print(f"Final number of df: {len(return_list)}")
    print("split_df() completed")
    print("="*50)

    return return_list


def split_train_val_test(dict_list, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split a list of dicts into training, validation, and test sets with random shuffling.
    
    Args:
        dict_list: list of dicts to split
        train_ratio: float, ratio of training data (default=0.6)
        val_ratio: float, ratio of validation data (default=0.2)
        test_ratio: float, ratio of test data (default=0.2)
    
    Returns:
        training_dict_list: list of dicts for training
        validation_dict_list: list of dicts for validation
        test_dict_list: list of dicts for testing
    """
    if not isinstance(dict_list, list):
        raise ValueError("Input must be a list")
    
    if len(dict_list) == 0:
        return [], [], []
    
    # Check if ratios sum to 1.0 (allow small floating point errors)
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.3f}")
    
    # Allow val_ratio=0 (no validation set)
    # But train_ratio and test_ratio must be > 0
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and test_ratio must be > 0")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be >= 0 and < 1")
    if train_ratio >= 1 or test_ratio >= 1:
        raise ValueError("train_ratio and test_ratio must be < 1")
    
    print("="*50)
    print("Starting 'split_train_val_test()'")
    print(f"Total length of dict: {len(dict_list)}")
    print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    
    # Random shuffle
    shuffled_indices = np.random.permutation(len(dict_list))
    
    # Calculate split indices
    train_idx = int(train_ratio * len(dict_list))
    val_idx = train_idx + int(val_ratio * len(dict_list)) if val_ratio > 0 else train_idx
    
    # Split into training, validation, and test
    training_dict_list = [dict_list[i] for i in shuffled_indices[:train_idx]]
    if val_ratio > 0:
        validation_dict_list = [dict_list[i] for i in shuffled_indices[train_idx:val_idx]]
    else:
        validation_dict_list = []  # Empty list if val_ratio=0
    test_dict_list = [dict_list[i] for i in shuffled_indices[val_idx:]]
    
    train_percent = (len(training_dict_list) / len(dict_list)) * 100
    val_percent = (len(validation_dict_list) / len(dict_list)) * 100
    test_percent = (len(test_dict_list) / len(dict_list)) * 100
    
    print(f"Number of train dict: {len(training_dict_list)} ({train_percent:.1f}%)")
    print(f"Number of val dict: {len(validation_dict_list)} ({val_percent:.1f}%)")
    print(f"Number of test dict: {len(test_dict_list)} ({test_percent:.1f}%)")
    print("split_train_val_test() completed")
    print("="*50)
    
    return training_dict_list, validation_dict_list, test_dict_list


def match_exact_keywords(key, keywords):
    """
    key가 예: 'udds2_1c_rest_25C_struct' 같이 _로 나눠진 경우,
    keywords 중 하나라도 정확히 단어로써(key의 토큰) 포함되어야 True.
    """
    # 언더스코어 단위 분할
    tokens = re.split(r'[_\s]', key)
    # 모든 keywords가 이 토큰 list에 "정확히 단어로써" 존재하는가?
    return any(word in tokens for word in keywords)


def extract_data(df_list, target_keyword, soc_lb=0.11, soc_ub=0.96):
    ret_list = []
    found_any = False


    for k, v in df_list.items():
        # 정확히 해당 단어가 분리된 토큰으로 존재하는지 확인
        keyword_match = match_exact_keywords(k, target_keyword)
        print(f"Key: {k}, keyword_match: {keyword_match}, type: {type(v)}")
        if keyword_match:
            if isinstance(v, dict):
                print("  Skipped (dict)")
                continue
            if hasattr(v, "columns") and ("soc_n" in v.columns) and ("Vref" in v.columns):
                mask = (v["soc_n"] >= soc_lb) & (v["soc_n"] <= soc_ub) & (v["Vref"] >= 2.5)
                filtered_df = v[mask].reset_index(drop=True)
                print(f"  Added! Rows: {len(filtered_df)}")
                ret_list.append(filtered_df)
                found_any = True
            else:
                print("  Columns missing, skipped")

    print(f"최종 target_list length: {len(ret_list)}")
    if not found_any:
        print("⚠️ 조건에 맞는 DataFrame이 없습니다. (key 토큰, 컬럼, 타입, 마스킹 등 모두 점검)")

    return ret_list


def extract_upper_envelope(data, window_size=7, percentile=97, column=None, 
                          smooth_window=None, smooth_method='moving_average'):
    """
    Extract upper envelope from time series data using sliding window percentile.
    Optionally apply additional smoothing to remove remaining pulses.
    
    Args:
        data: 1D numpy array, list, or pandas DataFrame
              If DataFrame, specify column name to extract
        window_size: int, size of sliding window for percentile calculation
        percentile: float, percentile value (0-100) for upper envelope
        column: str, column name if data is DataFrame (default: 'Vcorr')
        smooth_window: int or None, window size for additional smoothing (None = no smoothing)
        smooth_method: str, smoothing method ('moving_average', 'savgol', or 'butterworth')
    
    Returns:
        envelope: 1D numpy array, upper envelope of the input data
    
    Example:
        >>> # 방법 1: 배열 직접 사용
        >>> import numpy as np
        >>> data = np.array([1, 2, 5, 3, 4, 6, 2, 3])
        >>> envelope = extract_upper_envelope(data, window_size=3, percentile=90)
        >>> 
        >>> # 방법 2: DataFrame에서 Vcorr 컬럼 추출
        >>> envelope = extract_upper_envelope(df, window_size=15, percentile=95, column='Vcorr')
        >>> 
        >>> # 방법 3: 추가 smoothing 적용 (펄스 제거)
        >>> envelope = extract_upper_envelope(df, 
        ...                                   window_size=15, 
        ...                                   percentile=95,
        ...                                   column='Vcorr',
        ...                                   smooth_window=21, 
        ...                                   smooth_method='savgol')
        >>> 
        >>> # 방법 4: moving average smoothing
        >>> envelope = extract_upper_envelope(df,
        ...                                   window_size=25,
        ...                                   percentile=98,
        ...                                   smooth_window=31,
        ...                                   smooth_method='moving_average')
    """
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        if column is None:
            column = 'Vcorr'  # default column
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(data.columns)}")
        data = data[column].values
    
    # Convert to numpy array
    data = np.array(data)
    N = len(data)
    
    if N == 0:
        return np.array([])
    
    envelope = np.zeros(N)
    half = window_size  # 기존 코드와 동일하게 window_size를 half로 사용
    
    for i in range(N):
        start = max(0, i - half)
        end = min(N, i + half + 1)
        envelope[i] = np.percentile(data[start:end], percentile)
    
    # Apply additional smoothing if requested
    if smooth_window is not None and smooth_window > 1:
        if smooth_method == 'moving_average':
            # Moving average smoothing with better edge handling
            from scipy.ndimage import uniform_filter1d
            # Use 'reflect' mode for better edge handling (more uniform smoothing)
            # Apply multiple passes for smoother result (reduces step-wise artifacts)
            envelope = uniform_filter1d(envelope, size=smooth_window, mode='reflect')
            # Second pass with smaller window for fine smoothing
            if smooth_window >= 5:
                fine_window = max(3, smooth_window // 3)
                envelope = uniform_filter1d(envelope, size=fine_window, mode='reflect')
        elif smooth_method == 'savgol':
            # Savitzky-Golay filter (polynomial smoothing)
            # Ensure window is odd and not larger than data length
            smooth_window_odd = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            smooth_window_odd = min(smooth_window_odd, N if N % 2 == 1 else N - 1)
            if smooth_window_odd >= 3:
                # Use higher polynomial order for smoother result
                poly_order = min(3, smooth_window_odd - 1)  # polynomial order
                envelope = savgol_filter(envelope, smooth_window_odd, poly_order)
                # Apply second pass for extra smoothness (reduces step-wise artifacts)
                if smooth_window_odd >= 7:
                    second_window = max(5, smooth_window_odd - 4)
                    second_window = second_window if second_window % 2 == 1 else second_window + 1
                    second_window = min(second_window, N if N % 2 == 1 else N - 1)
                    if second_window >= 3:
                        envelope = savgol_filter(envelope, second_window, min(3, second_window - 1))
        elif smooth_method == 'butterworth':
            # Butterworth low-pass filter (frequency domain filter)
            from scipy.signal import butter, filtfilt
            # Normalize cutoff frequency (0.0 to 1.0, where 1.0 is Nyquist frequency)
            # smooth_window을 cutoff frequency로 해석: 작을수록 더 부드러움
            # 예: smooth_window=21이면 cutoff ≈ 1/21 ≈ 0.05 (매우 낮은 주파수만 통과)
            nyquist = 0.5
            cutoff = min(0.4, max(0.01, 1.0 / smooth_window))  # cutoff frequency
            b, a = butter(4, cutoff, btype='low', analog=False)  # 4th order Butterworth
            envelope = filtfilt(b, a, envelope)  # zero-phase filtering (forward + backward)
        else:
            raise ValueError(f"Unknown smooth_method: {smooth_method}. Use 'moving_average', 'savgol', or 'butterworth'")
    
    return envelope


def filter_Vcorr_single(df,
                        window_size=7,         # upper envelope window
                        percentile=97,         # upper envelope percentile
                        I_rest_threshold=0.02, # define rest
                        alpha=0.97,            # smoothing for non-rest only
                        Vcorr_col='Vcorr',
                        I_col='current',
                        smooth_window=None,     # additional smoothing window (None = no smoothing)
                        smooth_method='moving_average'):  # 'moving_average' or 'savgol'
    """
    Apply Vcorr regime-based filtering to a single DataFrame.
    
    Filtering algorithm:
    1. Extract upper envelope using sliding window percentile
    2. Apply regime-based filtering:
       - REST regime (|I| < threshold): use raw Vcorr (no smoothing)
       - DRIVING/PULSE regime (|I| >= threshold): use upper envelope + smoothing
    
    Args:
        df: pandas DataFrame with Vcorr and current columns
        window_size: int, upper envelope window size
        percentile: float, upper envelope percentile (0-100)
        I_rest_threshold: float, current threshold for rest detection
        alpha: float, smoothing parameter for non-rest only
        Vcorr_col: str, name of Vcorr column
        I_col: str, name of current column
    
    Returns:
        df_filtered: pandas DataFrame with filtered Vcorr column
    """
    df_copy = df.copy()
    
    if Vcorr_col not in df_copy.columns or I_col not in df_copy.columns:
        return df_copy
        
    Vcorr = df_copy[Vcorr_col].values
    I = df_copy[I_col].values
    N = len(Vcorr)
    
    # Step 1 — Upper envelope extraction
    Vupper = extract_upper_envelope(Vcorr, window_size, percentile, 
                                    smooth_window=smooth_window, 
                                    smooth_method=smooth_method)
    
    # Step 2 — Initialize Vslow
    Vslow = np.zeros(N)
    Vslow[0] = Vcorr[0]   # 초기값은 raw 그대로
    
    # Step 3 — Regime-based filtering
    for i in range(1, N):
        if abs(I[i]) < I_rest_threshold:
            # ===== REST 구간: smoothing 완전히 끄고 raw 사용 =====
            Vslow[i] = Vcorr[i]
        else:
            # ===== PULSE/DRIVING 구간: upper envelope + smoothing =====
            Vslow[i] = alpha * Vslow[i-1] + (1 - alpha) * Vupper[i]
    
    # Update Vcorr column with filtered values
    df_copy[Vcorr_col] = Vslow
    return df_copy


def filter_Vcorr_lpf_single(df,
                             I_rest_threshold=0.02,  # current threshold for rest detection
                             alpha=0.005,            # LPF smoothing parameter (small value for smooth transition)
                             Vcorr_col='Vcorr',
                             I_col='current'):
    """
    Apply hybrid rest + LPF filtering to a single DataFrame.
    
    Filtering algorithm:
    1. Find rest regions where |I| < I_rest_threshold (anchors)
    2. Use raw Vcorr at rest points (anchoring)
    3. Fill between rest points with LPF: Vslow(t) = (1-α)*Vslow(t-1) + α*Vcorr(t)
    4. Fill tail after last rest with LPF
    
    Args:
        df: pandas DataFrame with Vcorr and current columns
        I_rest_threshold: float, current threshold for rest detection
        alpha: float, LPF smoothing parameter (small value, e.g., 0.005 for smooth transition)
        Vcorr_col: str, name of Vcorr column
        I_col: str, name of current column
    
    Returns:
        df_filtered: pandas DataFrame with filtered Vcorr column
    """
    df_copy = df.copy()
    
    if Vcorr_col not in df_copy.columns or I_col not in df_copy.columns:
        return df_copy
        
    Vcorr = df_copy[Vcorr_col].values
    I = df_copy[I_col].values
    N = len(Vcorr)
    
    Vslow = np.zeros_like(Vcorr)
    
    # 1) rest index 추출 (앵커)
    rest_mask = np.abs(I) < I_rest_threshold
    rest_indices = np.where(rest_mask)[0]
    
    # 보호: rest가 없다면 그냥 LPF 반환
    if len(rest_indices) == 0:
        Vslow[0] = Vcorr[0]
        for t in range(1, N):
            Vslow[t] = (1 - alpha) * Vslow[t-1] + alpha * Vcorr[t]
        df_copy[Vcorr_col] = Vslow
        return df_copy
    
    # 2) 모든 rest 지점에서 raw Vcorr 사용 (앵커 고정, 손대지 않음)
    for rest_idx in rest_indices:
        Vslow[rest_idx] = Vcorr[rest_idx]
    
    # 3) 첫 rest 이전 구간 처리 (0부터 첫 rest까지)
    first_rest = rest_indices[0]
    if first_rest > 0:
        # 첫 rest 값에 수렴하도록 backward LPF
        Vslow[0] = Vcorr[0]
        for t in range(1, first_rest):
            # 목표: first_rest의 값에 가까워지도록
            target = Vcorr[first_rest]
            progress = t / first_rest  # 0에서 1로
            # 더 부드러운 전환을 위해 가중치 조정
            Vslow[t] = (1 - alpha) * Vslow[t-1] + alpha * (Vcorr[t] * (1 - progress * 0.5) + target * progress * 0.5)
    
    # 4) rest 구간 사이를 LPF로 채우기 (양방향으로 부드럽게 연결)
    for i in range(len(rest_indices) - 1):
        k0 = rest_indices[i]
        k1 = rest_indices[i+1]
        
        # k0와 k1은 이미 앵커로 고정됨 (raw Vcorr)
        # k0~k1 사이를 부드럽게 연결
        if k1 - k0 > 1:
            # 선형 보간으로 먼저 초기 추정값 생성 (매우 부드러운 베이스라인)
            linear_base = np.linspace(Vcorr[k0], Vcorr[k1], k1 - k0 + 1)
            
            # Forward pass: k0에서 시작해서 k1 방향으로
            Vslow_forward = np.zeros(k1 - k0 + 1)
            Vslow_forward[0] = Vcorr[k0]  # 앵커
            for t in range(k0 + 1, k1):
                idx = t - k0
                # 선형 보간 30% 사용 (30% 선형 보간 + 70% LPF)
                linear_val = linear_base[idx]
                Vslow_forward[idx] = (1 - alpha) * Vslow_forward[idx - 1] + alpha * (linear_val * 0.3 + Vcorr[t] * 0.7)
            
            # Backward pass: k1에서 시작해서 k0 방향으로
            Vslow_backward = np.zeros(k1 - k0 + 1)
            Vslow_backward[-1] = Vcorr[k1]  # 앵커
            for t in range(k1 - 1, k0, -1):
                idx = t - k0
                # 선형 보간 30% 사용
                linear_val = linear_base[idx]
                Vslow_backward[idx] = (1 - alpha) * Vslow_backward[idx + 1] + alpha * (linear_val * 0.3 + Vcorr[t] * 0.7)
            
            # Forward와 backward의 균등 평균 (더 부드러운 연결)
            for t in range(k0 + 1, k1):
                idx = t - k0
                # 선형 보간도 함께 고려
                linear_val = linear_base[idx]
                # 3-way 평균: forward, backward, linear
                Vslow[t] = 0.4 * Vslow_forward[idx] + 0.4 * Vslow_backward[idx] + 0.2 * linear_val
            
            # 추가 smoothing: 인접한 값들의 평균 (step-wise 제거)
            if k1 - k0 > 3:
                for t in range(k0 + 1, k1):
                    # 양쪽 인접 값과의 가중 평균
                    if t > k0 + 1 and t < k1 - 1:
                        Vslow[t] = 0.5 * Vslow[t] + 0.25 * Vslow[t-1] + 0.25 * Vslow[t+1]
                    elif t == k0 + 1:
                        Vslow[t] = 0.6 * Vslow[t] + 0.4 * Vslow[t+1]
                    elif t == k1 - 1:
                        Vslow[t] = 0.6 * Vslow[t] + 0.4 * Vslow[t-1]
    
    # 5) 마지막 rest 이후 tail 구간도 LPF
    last_rest = rest_indices[-1]
    if last_rest < N - 1:
        Vslow[last_rest] = Vcorr[last_rest]  # 앵커 유지
        for t in range(last_rest + 1, N):
            Vslow[t] = (1 - alpha) * Vslow[t-1] + alpha * Vcorr[t]
    
    # Update Vcorr column with filtered values
    df_copy[Vcorr_col] = Vslow
    return df_copy


def filter_Vcorr_lpf_regime(extracted_data,
                            I_rest_threshold=0.02,  # current threshold for rest detection
                            alpha=0.005,            # LPF smoothing parameter (small value for smooth transition)
                            Vcorr_col='Vcorr',
                            I_col='current'):
    """
    Apply hybrid rest + LPF filtering and merge drivingonly + restonly profiles.
    
    Process:
    1. Extract all drivingonly profiles (udds2_1c_drivingonly_25C_SOC_##)
    2. Apply hybrid rest + LPF to drivingonly profiles
    3. Extract all restonly profiles (udds2_1c_restonly_25C_SOC_##)
    4. Apply hybrid rest + LPF to restonly profiles
    5. Match drivingonly and restonly by SOC_## number
    6. Concatenate matched profiles (rest time starts from driving end time + 1s)
    7. Return as list of DataFrames
    
    Filtering algorithm:
    1. Find rest regions where |I| < I_rest_threshold (anchors)
    2. Use raw Vcorr at rest points (anchoring)
    3. Fill between rest points with LPF: Vslow(t) = (1-α)*Vslow(t-1) + α*Vcorr(t)
    4. Fill tail after last rest with LPF
    
    Input:
        extracted_data: dict with keys like 'udds2_1c_drivingonly_25C_SOC_53' and DataFrame values
        I_rest_threshold: float, current threshold for rest detection
        alpha: float, LPF smoothing parameter (small value, e.g., 0.005)
        Vcorr_col: str, name of Vcorr column
        I_col: str, name of current column
    
    Output:
        merged_list_filtered, merged_list_original: tuple of two lists
            - merged_list_filtered: list of DataFrames with hybrid rest + LPF applied
            - merged_list_original: list of DataFrames with only merge (no filtering)
        
        If input is a DataFrame, returns (filtered_df, original_df)
    """
    import re
    import pandas as pd
    
    # If input is a single DataFrame, apply filter and return both original and filtered
    if isinstance(extracted_data, pd.DataFrame):
        filtered_df = filter_Vcorr_lpf_single(extracted_data, I_rest_threshold, alpha, Vcorr_col, I_col)
        return filtered_df, extracted_data.copy()  # (filtered, original)
    
    if not isinstance(extracted_data, dict):
        raise ValueError("extracted_data must be a dictionary or DataFrame")
    
    # Pattern to match drivingonly and restonly profiles
    driving_pattern = r'udds\d+_1c_drivingonly_\d+C_SOC_(\d+)'
    rest_pattern = r'udds\d+_1c_restonly_\d+C_SOC_(\d+)'
    
    # Step 1 & 2: Extract drivingonly profiles (both original and filtered)
    driving_dict = {}  # key: SOC number, value: filtered DataFrame
    driving_dict_orig = {}  # key: SOC number, value: original DataFrame (no filtering)
    for key, df in extracted_data.items():
        match = re.search(driving_pattern, key)
        if match:
            soc_num = match.group(1)
            filtered_df = filter_Vcorr_lpf_single(df, I_rest_threshold, alpha, Vcorr_col, I_col)
            driving_dict[soc_num] = filtered_df
            driving_dict_orig[soc_num] = df.copy()  # 원본 저장
    
    # Step 3 & 4: Extract restonly profiles (both original and filtered)
    rest_dict = {}  # key: SOC number, value: filtered DataFrame
    rest_dict_orig = {}  # key: SOC number, value: original DataFrame (no filtering)
    for key, df in extracted_data.items():
        match = re.search(rest_pattern, key)
        if match:
            soc_num = match.group(1)
            filtered_df = filter_Vcorr_lpf_single(df, I_rest_threshold, alpha, Vcorr_col, I_col)
            rest_dict[soc_num] = filtered_df
            rest_dict_orig[soc_num] = df.copy()  # 원본 저장
    
    # Helper function to merge profiles
    def merge_profiles(driving_dict_merge, rest_dict_merge):
        merged_list = []
        time_col = 'time'
        
        # Find all SOC numbers that exist in both dictionaries
        common_soc = set(driving_dict_merge.keys()) & set(rest_dict_merge.keys())
        
        for soc_num in sorted(common_soc):
            driving_df = driving_dict_merge[soc_num].copy()
            rest_df = rest_dict_merge[soc_num].copy()
            
            # Ensure time column exists
            if time_col not in driving_df.columns or time_col not in rest_df.columns:
                print(f"Warning: 'time' column not found for SOC_{soc_num}. Skipping.")
                continue
            
            # Get the last time value from driving profile
            driving_end_time = driving_df[time_col].max()
            
            # Reset rest profile time to start from driving_end_time + 1
            rest_df[time_col] = rest_df[time_col] - rest_df[time_col].min() + driving_end_time + 1
            
            # Concatenate driving and rest DataFrames
            merged_df = pd.concat([driving_df, rest_df], ignore_index=True)
            merged_list.append(merged_df)
        
        # Add any driving-only profiles that don't have a matching rest profile
        driving_only_soc = set(driving_dict_merge.keys()) - set(rest_dict_merge.keys())
        for soc_num in sorted(driving_only_soc):
            merged_list.append(driving_dict_merge[soc_num].copy())
        
        # Add any rest-only profiles that don't have a matching driving profile
        rest_only_soc = set(rest_dict_merge.keys()) - set(driving_dict_merge.keys())
        for soc_num in sorted(rest_only_soc):
            merged_list.append(rest_dict_merge[soc_num].copy())
        
        return merged_list
    
    # Step 5 & 6: Merge filtered profiles (with 1st order LPF applied)
    merged_list_filtered = merge_profiles(driving_dict, rest_dict)
    
    # Step 7: Merge original profiles (no filtering, just merge)
    merged_list_original = merge_profiles(driving_dict_orig, rest_dict_orig)
    
    common_soc = set(driving_dict.keys()) & set(rest_dict.keys())
    driving_only_soc = set(driving_dict.keys()) - set(rest_dict.keys())
    rest_only_soc = set(rest_dict.keys()) - set(driving_dict.keys())
    
    print(f"Processed {len(common_soc)} matched pairs (driving+rest)")
    print(f"Added {len(driving_only_soc)} driving-only profiles")
    print(f"Added {len(rest_only_soc)} rest-only profiles")
    print(f"Total merged profiles (filtered): {len(merged_list_filtered)}")
    print(f"Total merged profiles (original): {len(merged_list_original)}")
    
    return merged_list_filtered, merged_list_original


def filter_Vcorr_regime(extracted_data,
                        window_size=7,         # upper envelope window
                        percentile=97,         # upper envelope percentile
                        I_rest_threshold=0.02, # define rest
                        alpha=0.97,            # smoothing for non-rest only
                        Vcorr_col='Vcorr',
                        I_col='current',
                        smooth_window=None,     # additional smoothing window (None = no smoothing)
                        smooth_method='moving_average'):  # 'moving_average' or 'savgol'
    """
    Filter Vcorr regime and merge drivingonly + restonly profiles.
    
    Process:
    1. Extract all drivingonly profiles (udds2_1c_drivingonly_25C_SOC_##)
    2. Apply filtering algorithm to drivingonly profiles
    3. Extract all restonly profiles (udds2_1c_restonly_25C_SOC_##)
    4. Apply filtering algorithm to restonly profiles
    5. Match drivingonly and restonly by SOC_## number
    6. Concatenate matched profiles (rest time starts from driving end time + 1s)
    7. Return as list of DataFrames
    
    Input:
        extracted_data: dict with keys like 'udds2_1c_drivingonly_25C_SOC_53' and DataFrame values
        window_size: int, upper envelope window
        percentile: float, upper envelope percentile
        I_rest_threshold: float, current threshold for rest detection (|I| < threshold = rest)
        alpha: float, smoothing parameter for non-rest only
        Vcorr_col: str, name of Vcorr column
        I_col: str, name of current column
    
    Output:
        merged_list_filtered, merged_list_original: tuple of two lists
            - merged_list_filtered: list of DataFrames with extract_upper_envelope applied
            - merged_list_original: list of DataFrames with only merge (no filtering)
        
        If input is a DataFrame, returns (filtered_df, original_df)
    """
    import re
    import pandas as pd
    
    # If input is a single DataFrame, apply filter and return both original and filtered
    if isinstance(extracted_data, pd.DataFrame):
        filtered_df = filter_Vcorr_single(extracted_data, window_size, percentile, 
                                          I_rest_threshold, alpha, Vcorr_col, I_col,
                                          smooth_window, smooth_method)
        return filtered_df, extracted_data.copy()  # (filtered, original)
    
    if not isinstance(extracted_data, dict):
        raise ValueError("extracted_data must be a dictionary or DataFrame")
    
    # Pattern to match drivingonly and restonly profiles
    driving_pattern = r'udds\d+_1c_drivingonly_\d+C_SOC_(\d+)'
    rest_pattern = r'udds\d+_1c_restonly_\d+C_SOC_(\d+)'
    
    # Step 1 & 2: Extract drivingonly profiles (both original and filtered)
    driving_dict = {}  # key: SOC number, value: filtered DataFrame
    driving_dict_orig = {}  # key: SOC number, value: original DataFrame (no filtering)
    for key, df in extracted_data.items():
        match = re.search(driving_pattern, key)
        if match:
            soc_num = match.group(1)
            filtered_df = filter_Vcorr_single(df, window_size, percentile, I_rest_threshold,
                                              alpha, Vcorr_col, I_col,
                                              smooth_window, smooth_method)
            driving_dict[soc_num] = filtered_df
            driving_dict_orig[soc_num] = df.copy()  # 원본 저장
    
    # Step 3 & 4: Extract restonly profiles (both original and filtered)
    rest_dict = {}  # key: SOC number, value: filtered DataFrame
    rest_dict_orig = {}  # key: SOC number, value: original DataFrame (no filtering)
    for key, df in extracted_data.items():
        match = re.search(rest_pattern, key)
        if match:
            soc_num = match.group(1)
            filtered_df = filter_Vcorr_single(df, window_size, percentile, I_rest_threshold,
                                              alpha, Vcorr_col, I_col,
                                              smooth_window, smooth_method)
            rest_dict[soc_num] = filtered_df
            rest_dict_orig[soc_num] = df.copy()  # 원본 저장
    
    # Helper function to merge profiles
    def merge_profiles(driving_dict_merge, rest_dict_merge):
        merged_list = []
        time_col = 'time'
        
        # Find all SOC numbers that exist in both dictionaries
        common_soc = set(driving_dict_merge.keys()) & set(rest_dict_merge.keys())
        
        for soc_num in sorted(common_soc):
            driving_df = driving_dict_merge[soc_num].copy()
            rest_df = rest_dict_merge[soc_num].copy()
            
            # Ensure time column exists
            if time_col not in driving_df.columns or time_col not in rest_df.columns:
                print(f"Warning: 'time' column not found for SOC_{soc_num}. Skipping.")
                continue
            
            # Get the last time value from driving profile
            driving_end_time = driving_df[time_col].max()
            
            # Reset rest profile time to start from driving_end_time + 1
            rest_df[time_col] = rest_df[time_col] - rest_df[time_col].min() + driving_end_time + 1
            
            # Concatenate driving and rest DataFrames
            merged_df = pd.concat([driving_df, rest_df], ignore_index=True)
            merged_list.append(merged_df)
        
        # Add any driving-only profiles that don't have a matching rest profile
        driving_only_soc = set(driving_dict_merge.keys()) - set(rest_dict_merge.keys())
        for soc_num in sorted(driving_only_soc):
            merged_list.append(driving_dict_merge[soc_num].copy())
        
        # Add any rest-only profiles that don't have a matching driving profile
        rest_only_soc = set(rest_dict_merge.keys()) - set(driving_dict_merge.keys())
        for soc_num in sorted(rest_only_soc):
            merged_list.append(rest_dict_merge[soc_num].copy())
        
        return merged_list
    
    # Step 5 & 6: Merge filtered profiles (with extract_upper_envelope applied)
    merged_list_filtered = merge_profiles(driving_dict, rest_dict)
    
    # Step 7: Merge original profiles (no filtering, just merge)
    merged_list_original = merge_profiles(driving_dict_orig, rest_dict_orig)
    
    common_soc = set(driving_dict.keys()) & set(rest_dict.keys())
    driving_only_soc = set(driving_dict.keys()) - set(rest_dict.keys())
    rest_only_soc = set(rest_dict.keys()) - set(driving_dict.keys())
    
    print(f"Processed {len(common_soc)} matched pairs (driving+rest)")
    print(f"Added {len(driving_only_soc)} driving-only profiles")
    print(f"Added {len(rest_only_soc)} rest-only profiles")
    print(f"Total merged profiles (filtered): {len(merged_list_filtered)}")
    print(f"Total merged profiles (original): {len(merged_list_original)}")
    
    return merged_list_filtered, merged_list_original








def merge_udds(extracted_data):
    """
    Merge drivingonly and restonly DataFrames from dict with matching UDDS, temperature, and SOC levels.
    
    Args:
        extracted_data: dict with keys like 'udds2_1c_drivingonly_25C_SOC_53' and DataFrame values
    
    Returns:
        merged_df_list: list of pandas DataFrame, merged profiles (drivingonly + restonly)
    """
    # Helper function to extract profile info from name
    def extract_profile_info(name):
        """
        Extract UDDS, temperature, and SOC from profile name
        Example: 'udds2_1c_drivingonly_25C_SOC_53' -> ('udds2', '1c', '25C', '53', 'drivingonly')
        """
        # Pattern: udds{number}_{rate}c_{type}_{temp}C_SOC_{soc}
        pattern = r'(udds\d+)_(\d+c)_(drivingonly|restonly)_(\d+C)_SOC_(\d+)'
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            udds = match.group(1)
            rate = match.group(2)
            profile_type = match.group(3)
            temp = match.group(4)
            soc = match.group(5)
            return (udds, rate, temp, soc, profile_type)
        return None
    
    # Create dictionaries to store DataFrames by profile key
    driving_dict = {}
    rest_dict = {}
    
    # Process extracted_data dict
    for key, df in extracted_data.items():
        info = extract_profile_info(key)
        if info:
            udds, rate, temp, soc, profile_type = info
            profile_key = (udds, rate, temp, soc)  # Key without drivingonly/restonly
            
            if profile_type.lower() == 'drivingonly':
                driving_dict[profile_key] = df.copy()
            elif profile_type.lower() == 'restonly':
                rest_dict[profile_key] = df.copy()
    
    # Merge matching profiles
    merged_df_list = []
    time_col = 'time'
    
    # Find all unique keys that exist in both dictionaries
    common_keys = set(driving_dict.keys()) & set(rest_dict.keys())
    
    for key in common_keys:
        udds, rate, temp, soc = key
        driving_df = driving_dict[key].copy()
        rest_df = rest_dict[key].copy()
        
        # Ensure time column exists
        if time_col not in driving_df.columns or time_col not in rest_df.columns:
            print(f"Warning: 'time' column not found for SOC_{soc}. Skipping.")
            continue
        
        # Get the last time value from driving profile
        driving_end_time = driving_df[time_col].max()
        
        # Reset rest profile time to start from driving_end_time + 1
        rest_df[time_col] = rest_df[time_col] - rest_df[time_col].min() + driving_end_time + 1
        
        # Concatenate driving and rest DataFrames
        merged_df = pd.concat([driving_df, rest_df], ignore_index=True)
        
        merged_df_list.append(merged_df)
    
    # Add any driving-only profiles that don't have a matching rest profile
    driving_only_keys = set(driving_dict.keys()) - set(rest_dict.keys())
    for key in driving_only_keys:
        merged_df_list.append(driving_dict[key].copy())
    
    # Add any rest-only profiles that don't have a matching driving profile
    rest_only_keys = set(rest_dict.keys()) - set(driving_dict.keys())
    for key in rest_only_keys:
        merged_df_list.append(rest_dict[key].copy())
    
    print(f"Merged {len(common_keys)} pairs of driving+rest profiles")
    print(f"Added {len(driving_only_keys)} driving-only profiles")
    print(f"Added {len(rest_only_keys)} rest-only profiles")
    print(f"Total merged profiles: {len(merged_df_list)}")
    
    return merged_df_list


def plot_merged_udds(dict_list1, dict_list2, figsize=(15, 10)):
    """
    Plot two lists of dicts side by side, matching by index.
    
    Args:
        dict_list1: list of dicts, each dict should contain 'time' and 'Vcorr' keys
        dict_list2: list of dicts, each dict should contain 'time' and 'Vcorr' keys
                   (should have same length as dict_list1)
        figsize: tuple, figure size (width, height)
    
    Returns:
        None (displays plot)
    """
    import matplotlib.pyplot as plt
    
    if len(dict_list1) != len(dict_list2):
        raise ValueError(f"Two lists must have the same length. Got {len(dict_list1)} and {len(dict_list2)}")
    
    n_plots = len(dict_list1)
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx in range(n_plots):
        ax = axes[idx]
        dict1 = dict_list1[idx]
        dict2 = dict_list2[idx]
        
        # Extract time and Vcorr from first dict
        if 'time' in dict1 and 'Vcorr' in dict1:
            time1 = dict1['time']
            vcorr1 = dict1['Vcorr']
            
            # Convert to numpy array if needed
            if isinstance(time1, (list, tuple)):
                time1 = np.array(time1)
            if isinstance(vcorr1, (list, tuple)):
                vcorr1 = np.array(vcorr1)
            
            ax.plot(time1, vcorr1, label='List 1', alpha=0.7, linewidth=1.5)
        
        # Extract time and Vcorr from second dict
        if 'time' in dict2 and 'Vcorr' in dict2:
            time2 = dict2['time']
            vcorr2 = dict2['Vcorr']
            
            # Convert to numpy array if needed
            if isinstance(time2, (list, tuple)):
                time2 = np.array(time2)
            if isinstance(vcorr2, (list, tuple)):
                vcorr2 = np.array(vcorr2)
            
            ax.plot(time2, vcorr2, label='List 2', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Vcorr')
        ax.set_title(f'Index {idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_filtered_data(filtered_data, original_data=None, figsize=(15, 10), max_plots=None):
    """
    Plot filtered data from filter_Vcorr_regime function result, optionally with original data.
    
    Args:
        filtered_data: list of DataFrames (result from filter_Vcorr_regime function)
        original_data: dict with keys like 'udds2_1c_drivingonly_25C_SOC_##' and DataFrame values
                      (optional, if provided, original Vcorr will be plotted together)
        figsize: tuple, figure size (width, height)
        max_plots: int or None, maximum number of plots to show (None = show all)
    
    Returns:
        fig: matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import re
    
    if not isinstance(filtered_data, list):
        raise ValueError("filtered_data must be a list of DataFrames")
    
    n_plots = len(filtered_data)
    if max_plots is not None:
        n_plots = min(n_plots, max_plots)
        filtered_data = filtered_data[:n_plots]
    
    if n_plots == 0:
        print("No data to plot")
        return None
    
    # Extract original drivingonly data if provided
    original_dict = {}
    if original_data is not None and isinstance(original_data, dict):
        driving_pattern = r'udds\d+_1c_drivingonly_\d+C_SOC_(\d+)'
        for key, df in original_data.items():
            match = re.search(driving_pattern, key)
            if match:
                soc_num = match.group(1)
                original_dict[soc_num] = df
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # For each filtered DataFrame, try to find matching original by SOC
    # filter_Vcorr_regime returns data sorted by SOC, so we match by index
    for idx, filtered_df in enumerate(filtered_data):
        ax = axes[idx]
        
        # Hold on to plot multiple lines on same axes
        ax.hold = True  # For compatibility, though matplotlib handles this automatically
        
        if not isinstance(filtered_df, pd.DataFrame):
            ax.text(0.5, 0.5, f'Not a DataFrame\nItem {idx}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Item {idx} (Invalid)')
            continue
        
        if 'time' not in filtered_df.columns or 'Vcorr' not in filtered_df.columns:
            ax.text(0.5, 0.5, f'Missing columns\nItem {idx}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Item {idx} (No data)')
            continue
        
        # Plot original data first (if available)
        plot_original = False
        if original_data is not None and len(original_dict) > 0:
            # Get SOC numbers sorted (same order as filter_Vcorr_regime output)
            soc_nums = sorted(original_dict.keys())
            if idx < len(soc_nums):
                soc_num = soc_nums[idx]
                original_df = original_dict[soc_num]
                
                if isinstance(original_df, pd.DataFrame):
                    if 'time' in original_df.columns and 'Vcorr' in original_df.columns:
                        time_original = original_df['time'].values
                        vcorr_original = original_df['Vcorr'].values
                        
                        # Plot original drivingonly data
                        ax.plot(time_original, vcorr_original, 
                               label='Original (drivingonly)', alpha=0.8, linewidth=2.0, 
                               color='red', linestyle='--', marker='o', markersize=2)
                        plot_original = True
        
        # Plot filtered data (on same axes)
        time_filtered = filtered_df['time'].values
        vcorr_filtered = filtered_df['Vcorr'].values
        ax.plot(time_filtered, vcorr_filtered, 
               label='Filtered (driving+rest)', alpha=0.8, linewidth=2.0, color='blue')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Vcorr (V)')
        if plot_original:
            ax.set_title(f'Item {idx} - SOC_{soc_num if "soc_num" in locals() else "?"}')
        else:
            ax.set_title(f'Item {idx}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig
