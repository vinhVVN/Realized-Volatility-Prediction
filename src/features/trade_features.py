import numpy as np
import pandas as pd
from typing import List

def realized_volatility(series: pd.Series) -> float:
    """Tính toán realized volatility (căn bậc 2 của tổng bình phương log returns)."""
    return np.sqrt(np.sum(series**2))

def get_trade_features(df: pd.DataFrame, window_sizes: List[int] = [0, 150, 300, 450]) -> pd.DataFrame:
    """
    Tính toán các đặc trưng từ trade data với temporal windowing sử dụng thuần Pandas vectorization.
    
    Args:
        df (pd.DataFrame): DataFrame của trade data raw.
        window_sizes (List[int]): Danh sách các mốc thời gian lấy sub-window.
                                  Giá trị 0 là lấy toàn bộ window 600s.
                                  
    Returns:
        pd.DataFrame: DataFrame chứa các features đã được aggregated và flatten columns.
    """
    # 1. Tính toán log return ở cấp độ hàng (row-level) bằng vectorization
    df['log_return'] = np.log(df['price']).groupby(df['time_id']).diff()
    
    # 2. Định nghĩa từ điển aggregation
    create_feature_dict = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': ['count'],
        'size': [np.sum],
        'order_count': [np.mean]
    }
    
    df_feature = pd.DataFrame()
    
    # 3. Tạo aggregation cho từng temporal window
    for window in window_sizes:
        df_window = df[df['seconds_in_bucket'] >= window]
        df_agg = df_window.groupby('time_id').agg(create_feature_dict).reset_index()
        
        # Flatten columns: 'size', 'sum' -> 'size_sum'
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg = df_agg.rename(columns={'time_id_': 'time_id'})
        
        # Hậu tố window size nếu không phải là full window (0)
        suffix = f"_{window}" if window > 0 else ""
        
        if df_feature.empty:
            df_feature = df_agg
        else:
            # Gắn suffix cho các cột không phải time_id
            df_agg = df_agg.rename(columns={col: f"{col}{suffix}" for col in df_agg.columns if col != 'time_id'})
            df_feature = df_feature.merge(df_agg, how='left', on='time_id')
            
    # 4. Thêm tiền tố "trade_" vào tất cả các cột để dễ phân biệt với book
    df_feature = df_feature.rename(columns={col: f"trade_{col}" for col in df_feature.columns if col != 'time_id'})
    
    return df_feature
