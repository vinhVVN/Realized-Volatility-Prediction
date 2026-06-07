import numpy as np
import pandas as pd

def estimate_tick_size_and_real_price(df_book_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ước lượng tick_size và tính toán real_price cho mỗi stock_id dựa trên dữ liệu raw order book.
    
    Thuật toán: Tìm khoảng chênh lệch nhỏ nhất > 0 giữa các mức giá xuất hiện trong order book.
    
    Args:
        df_book_raw (pd.DataFrame): Dữ liệu book raw, cần chứa cột 'stock_id' và các cột giá 
                                    (ask_price1, ask_price2, bid_price1, bid_price2).
        
    Returns:
        pd.DataFrame: DataFrame map mỗi 'stock_id' với 'tick_size' và 'real_price' tương ứng.
    """
    tick_sizes = []
    real_prices = []
    stock_ids = df_book_raw['stock_id'].unique()
    
    for stock_id in stock_ids:
        df_stock = df_book_raw[df_book_raw['stock_id'] == stock_id]
        
        # Gộp tất cả các mức giá thành 1 mảng 1D
        prices = np.concatenate([
            df_stock['ask_price1'].unique(),
            df_stock['ask_price2'].unique(),
            df_stock['bid_price1'].unique(),
            df_stock['bid_price2'].unique()
        ])
        
        # Lấy giá trị unique và sắp xếp tăng dần
        unique_prices = np.sort(np.unique(prices))
        
        # Tính sai phân (difference) giữa các mức giá liền kề
        diffs = np.diff(unique_prices)
        
        # Lấy sai phân nhỏ nhất > 0 làm ước lượng cho tick_size
        valid_diffs = diffs[diffs > 0]
        if len(valid_diffs) > 0:
            tick_size = np.min(valid_diffs)
        else:
            tick_size = np.nan
            
        # Tính real_price theo công thức từ solution
        real_price = 0.01 / tick_size if tick_size and not np.isnan(tick_size) else np.nan
        
        tick_sizes.append(tick_size)
        real_prices.append(real_price)
        
    return pd.DataFrame({
        'stock_id': stock_ids,
        'tick_size': tick_sizes,
        'real_price': real_prices
    })
