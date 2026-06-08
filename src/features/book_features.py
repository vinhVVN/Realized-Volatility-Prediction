import numpy as np
import pandas as pd
from typing import List

def realized_volatility(series: pd.Series) -> float:
    """Tính toán realized volatility (căn bậc 2 của tổng bình phương log returns)."""
    return np.sqrt(np.sum(series**2))

def get_book_features(df: pd.DataFrame, window_sizes: List[int] = [0, 150, 300, 450]) -> pd.DataFrame:
    """
    Tính toán các đặc trưng từ book data với temporal windowing sử dụng thuần Pandas vectorization.
    
    Args:
        df (pd.DataFrame): DataFrame của book data raw.
        window_sizes (List[int]): Danh sách các mốc thời gian lấy sub-window (ví dụ: 150 là từ giây 150 đến cuối).
                                  Giá trị 0 là lấy toàn bộ window 600s.
                                  
    Returns:
        pd.DataFrame: DataFrame chứa các features đã được aggregated và flatten columns.
    """
    # 1. Tính toán các đặc trưng cơ bản ở cấp độ hàng (row-level)
    df['wap1'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['wap2'] = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    
    # Tính log return bằng vectorization (nhanh hơn apply)
    df['log_return1'] = np.log(df['wap1']).groupby([df['stock_id'], df['time_id']]).diff()
    df['log_return2'] = np.log(df['wap2']).groupby([df['stock_id'], df['time_id']]).diff()
    
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # 2. Định nghĩa từ điển aggregation
    create_feature_dict = {
        'wap1': ['sum', 'mean', 'std'],
        'wap2': ['sum', 'mean', 'std'],
        'log_return1': ['sum', realized_volatility, 'mean', 'std'],
        'log_return2': ['sum', realized_volatility, 'mean', 'std'],
        'wap_balance': ['sum', 'mean', 'std'],
        'price_spread': ['sum', 'mean', 'std'],
        'bid_spread': ['sum', 'mean', 'std'],
        'ask_spread': ['sum', 'mean', 'std'],
        'total_volume': ['sum', 'mean', 'std'],
        'volume_imbalance': ['sum', 'mean', 'std']
    }
    
    df_feature = pd.DataFrame()
    
    # 3. Tạo aggregation cho từng temporal window
    for window in window_sizes:
        df_window = df[df['seconds_in_bucket'] >= window]
        df_agg = df_window.groupby(['stock_id', 'time_id']).agg(create_feature_dict).reset_index()
        
        # Flatten columns: 'wap1', 'mean' -> 'wap1_mean'
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg = df_agg.rename(columns={'stock_id_': 'stock_id', 'time_id_': 'time_id'})
        
        # Hậu tố window size nếu không phải là full window (0)
        suffix = f"_{window}" if window > 0 else ""
        
        if df_feature.empty:
            df_feature = df_agg
            # Tính count snapshot (số lượng snapshot trong window hiện tại)
            count_s = df_window.groupby(['stock_id', 'time_id']).size().reset_index(name='snapshot_count')
            df_feature = df_feature.merge(count_s, on=['stock_id', 'time_id'], how='left')
        else:
            # Gắn suffix cho các cột không phải time_id và stock_id
            df_agg = df_agg.rename(columns={col: f"{col}{suffix}" for col in df_agg.columns if col not in ['stock_id', 'time_id']})
            df_feature = df_feature.merge(df_agg, how='left', on=['stock_id', 'time_id'])
            # Thêm cột snapshot count cho sub-window
            count_s = df_window.groupby(['stock_id', 'time_id']).size().reset_index(name=f'snapshot_count{suffix}')
            df_feature = df_feature.merge(count_s, how='left', on=['stock_id', 'time_id'])
            
    # 4. Thêm tiền tố "book_" vào tất cả các cột để dễ phân biệt với trade
    df_feature = df_feature.rename(columns={col: f"book_{col}" for col in df_feature.columns if col not in ['stock_id', 'time_id']})
    
    return df_feature
