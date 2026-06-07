import os
import gc
import yaml
import pandas as pd

from src.data.data_loader import DataLoader, DataBlock
from src.features.book_features import get_book_features
from src.features.trade_features import get_trade_features
from src.features.tick_features import estimate_tick_size_and_real_price
from src.features.misc_features import calculate_tau_features, apply_skew_correction
from src.features.rank_features import apply_rank_normalization
from src.features.nn_features import TimeIdNeighbors, StockIdNeighbors
from src.utils.logger import get_logger
from src.utils.timer import timer

logger = get_logger("feature_pipeline")

def build_features(df_train: pd.DataFrame, data_dir: str, block: DataBlock, config_path: str = 'configs/feature_config.yaml') -> pd.DataFrame:
    """
    Hàm master orchestrate toàn bộ feature engineering pipeline.
    
    Quy trình:
      a) Đọc dữ liệu (Book, Trade)
      b) Base features (WAP, spreads, returns) & Tick Size
      c) Ghép bảng, Tau, Skew Correction, Rank Normalization
      d) K-NN features với đa dạng metrics
      e) Hợp nhất thành 1 file duy nhất hoàn thiện.
    
    Args:
        df_train (pd.DataFrame): DataFrame nền tảng chứa time_id, stock_id (từ train.csv hoặc test.csv).
        data_dir (str): Thư mục gốc chứa các file parquet.
        block (DataBlock): Dữ liệu lấy từ TRAIN, TEST hay BOTH.
        config_path (str): Đường dẫn config YAML.
        
    Returns:
        pd.DataFrame: Bảng features hoàn chỉnh và gọn gàng.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    window_sizes = config.get('window_sizes', [450, 300, 150])
    if 0 not in window_sizes:
        window_sizes = [0] + window_sizes # 0 tương đương full 600s
        
    loader = DataLoader(data_dir=data_dir)
    
    # Lấy danh sách stock_id từ tập df_train truyền vào để filter, CHỐNG OOM CỰC KỲ HIỆU QUẢ!
    stock_ids = None
    if 'stock_id' in df_train.columns:
        stock_ids = df_train['stock_id'].unique().tolist()
        logger.info(f"Đã phát hiện {len(stock_ids)} unique stock_id trong df_train. Tiến hành filter dữ liệu đọc vào RAM.")
    
    with timer("a) Đọc dữ liệu Book và Trade song song", logger):
        df_book = loader.load_parquet_parallel('book', block=block, stock_ids=stock_ids)
        df_trade = loader.load_parquet_parallel('trade', block=block, stock_ids=stock_ids)
        
        if df_book.empty and df_trade.empty:
            logger.warning("Không lấy được bất kỳ data nào. Trả về dataframe ban đầu.")
            return df_train
        
    with timer("b) Tính toán Base Features và Tick Size", logger):
        # Tick size cần dataframe raw book
        df_tick = estimate_tick_size_and_real_price(df_book)
        
        df_book_features = get_book_features(df_book, window_sizes)
        del df_book
        gc.collect() # Ép giải phóng bộ nhớ
        
        df_trade_features = get_trade_features(df_trade, window_sizes)
        del df_trade
        gc.collect()
        
    with timer("c) Merge dữ liệu, tính Tau, Skew Correction và Rank Normalization", logger):
        # Trộn features vào móng df_train
        df = df_train.merge(df_book_features, on=['time_id', 'stock_id'], how='left')
        df = df.merge(df_trade_features, on=['time_id', 'stock_id'], how='left')
        df = df.merge(df_tick, on='stock_id', how='left')
        
        del df_book_features, df_trade_features, df_tick
        gc.collect()
        
        df = calculate_tau_features(df)
        df = apply_skew_correction(df)
        
        cols_to_rank = [
            'trade_order_count_mean', 'trade_size_sum', 
            'book_total_volume_sum', 'book_volume_imbalance_sum'
        ]
        df = apply_rank_normalization(df, cols_to_rank)
        
    with timer("d) Xây dựng K-NN Features (Core Innovation)", logger):
        nn_config = config.get('nn_features', {})
        n_neighbors_max = nn_config.get('n_neighbors_max', 80)
        time_windows = nn_config.get('time_windows', [3, 5, 10, 20, 40])
        vol_windows = nn_config.get('vol_windows', [2, 3, 5, 10, 20, 40])
        stock_windows = nn_config.get('stock_windows', [10, 20, 40])
        metrics_config = nn_config.get('metrics', [])
        
        vol_targets = [
            'book_log_return1_realized_volatility',
            'book_log_return2_realized_volatility',
            'trade_log_return_realized_volatility'
        ]
        
        for metric_cfg in metrics_config:
            name = metric_cfg['name']
            type_nn = metric_cfg['type']
            pivot = metric_cfg['pivot']
            distance = metric_cfg['distance']
            p_val = metric_cfg.get('p', 2)
            
            if pivot == 'volatility':
                pivot_col = 'book_log_return1_realized_volatility'
                target_cols = vol_targets
                n_wins = vol_windows
            elif pivot == 'real_price':
                pivot_col = 'real_price'
                target_cols = ['real_price']
                n_wins = time_windows if type_nn == 'time' else stock_windows
            elif pivot == 'total_volume':
                pivot_col = 'book_total_volume_sum'
                target_cols = ['book_total_volume_sum']
                n_wins = time_windows if type_nn == 'time' else stock_windows
            else:
                continue
                
            logger.info(f"   => Chạy thuật toán NN: {name} (type: {type_nn}, dist: {distance})")
            
            if type_nn == 'time':
                nn = TimeIdNeighbors(metric=distance, p=p_val, n_neighbors_max=n_neighbors_max)
            else:
                nn = StockIdNeighbors(metric=distance, p=p_val, n_neighbors_max=n_neighbors_max)
                
            df_nn = nn.generate_features(df, pivot_col=pivot_col, target_cols=target_cols, 
                                         n_windows=n_wins, prefix=name)
                                         
            df = df.merge(df_nn.reset_index(), on=['time_id', 'stock_id'], how='left')
            del df_nn, nn
            gc.collect()
            
    with timer("e) Final Cleanup", logger):
        # Đảm bảo không có cột nào bị lặp lại tên
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f"Feature Pipeline hoàn tất. Kích thước output: {df.shape}")
        
    return df
