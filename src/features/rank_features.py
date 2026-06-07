import pandas as pd
from typing import List

def apply_rank_normalization(df: pd.DataFrame, columns_to_rank: List[str]) -> pd.DataFrame:
    """
    Thực hiện Rank Normalization cho các features có phân phối thay đổi (non-stationary).
    Ví dụ: total_volume, order_count.
    
    Nhóm theo 'time_id' và rank qua toàn bộ các 'stock_id'. Điều này giúp model hiểu
    tính thanh khoản/mật độ giao dịch tương đối của cổ phiếu so với toàn thị trường 
    ngay tại thời điểm đó.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu tổng hợp (cần có cột 'time_id').
        columns_to_rank (List[str]): Danh sách tên các cột cần áp dụng rank.
        
    Returns:
        pd.DataFrame: DataFrame sau khi thay thế (in-place) các giá trị gốc bằng rank tương đối.
    """
    for col in columns_to_rank:
        if col in df.columns:
            # Rank tương đối theo time_id
            df[col] = df.groupby('time_id')[col].rank()
            
    return df
