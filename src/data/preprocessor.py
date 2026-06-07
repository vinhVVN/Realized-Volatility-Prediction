import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("preprocessor")

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Giảm dung lượng bộ nhớ của DataFrame bằng cách chuyển đổi (downcast) các kiểu dữ liệu
    sang kích thước nhỏ nhất có thể mà không làm mất độ chính xác.
    
    Args:
        df (pd.DataFrame): DataFrame cần tối ưu.
        
    Returns:
        pd.DataFrame: DataFrame sau khi đã được tối ưu bộ nhớ.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                # Tránh dùng float16 vì nó không tương thích tốt trên một số mô hình/cấu trúc dữ liệu
                # nên ta quy chuẩn về ít nhất float32
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
    logger.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df
