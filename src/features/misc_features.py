import numpy as np
import pandas as pd
from typing import List, Optional

def calculate_tau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán các đặc trưng tau: Tau_trade và Tau_book.
    Tau đo lường căn bậc 2 của nghịch đảo số lượng quan sát.
    
    Args:
        df (pd.DataFrame): DataFrame tổng hợp (sau khi đã merge book và trade features).
                           Yêu cầu tồn tại các cột như 'trade_seconds_in_bucket_count' và 'book_snapshot_count'.
                           
    Returns:
        pd.DataFrame: DataFrame có thêm các cột tau.
    """
    # Tính Tau cho full window
    if 'trade_seconds_in_bucket_count' in df.columns:
        df['tau_trade'] = np.sqrt(1 / df['trade_seconds_in_bucket_count'].replace(0, np.nan))
    
    if 'book_snapshot_count' in df.columns:
        df['tau_book'] = np.sqrt(1 / df['book_snapshot_count'].replace(0, np.nan))
        
    # Tính Tau cho các sub-windows được tạo ra từ temporal windowing
    # Lặp trên tập columns hiện tại (tránh lỗi dict size changed)
    for col in list(df.columns):
        if col.startswith('trade_seconds_in_bucket_count_'):
            suffix = col.replace('trade_seconds_in_bucket_count', '')
            df[f'tau_trade{suffix}'] = np.sqrt(1 / df[col].replace(0, np.nan))
        elif col.startswith('book_snapshot_count_'):
            suffix = col.replace('book_snapshot_count', '')
            df[f'tau_book{suffix}'] = np.sqrt(1 / df[col].replace(0, np.nan))
            
    return df

def apply_skew_correction(df: pd.DataFrame, columns_to_log: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Sử dụng np.log1p() để chuẩn hóa (skew correction) các cột có phân phối quá lệch,
    đặc biệt là các metrics về khối lượng.
    
    Args:
        df (pd.DataFrame): DataFrame dữ liệu tổng hợp.
        columns_to_log (List[str], optional): Các từ khóa hoặc tên cột cần log.
                                              Mặc định sẽ quét các keyword: 'size_sum', 'total_volume_sum', 'volume_imbalance_sum'.
                                              
    Returns:
        pd.DataFrame: DataFrame đã được log transform in-place.
    """
    if columns_to_log is None:
        columns_to_log = ['size_sum', 'total_volume_sum', 'volume_imbalance_sum']
        
    for col in df.columns:
        # Kiểm tra nếu tên cột chứa bất kỳ từ khóa nào cần log
        if any(keyword in col for keyword in columns_to_log):
            # Tránh lỗi log với giá trị âm, chặn dưới là 0
            df[col] = np.log1p(np.maximum(0, df[col]))
            
    return df
