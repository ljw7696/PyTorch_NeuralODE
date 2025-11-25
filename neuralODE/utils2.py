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
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5




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
        st = 0  # 시작 인덱스
        df_len = len(df)

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
                break  # 쪼갤 수 없음
            
            # valid_indices는 times[st:]의 상대 인덱스이므로, 원래 인덱스로 변환
            end_idx = st + valid_indices[-1]

            # 너무 작은 window는 버림 (예: window의 80% 미만)
            actual_window_time = times[end_idx] - curr_time
            if actual_window_time < window_seconds * 0.8:
                break

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
    
    # Check if each ratio is between 0 and 1 (inclusive)
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if val_ratio < 0 or val_ratio > 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if test_ratio < 0 or test_ratio > 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    
    # At least one ratio must be > 0 (cannot all be 0)
    if train_ratio == 0 and val_ratio == 0 and test_ratio == 0:
        raise ValueError("At least one ratio must be > 0")
    
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