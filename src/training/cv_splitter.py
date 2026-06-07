import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from typing import Tuple, Generator
from src.utils.logger import get_logger

logger = get_logger("cv_splitter")

class TimeSeriesCVSplitter:
    """
    Quản lý khôi phục trật tự thời gian (Time-ID) bị làm xáo trộn bằng t-SNE 
    và thực hiện TimeSeries Group K-Fold Cross Validation để chống data leakage.
    """
    def __init__(self, n_splits: int = 4, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.recovered_time_order = None
        
    def reverse_engineer_time_order(self, df_train: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Khôi phục trật tự thời gian thực bằng cách nhúng t-SNE 1D trên ma trận real_price.
        
        Args:
            df_train (pd.DataFrame): Dữ liệu gốc (cần có cột time_id).
            df_features (pd.DataFrame): Dữ liệu chứa features (cần có time_id, stock_id, real_price).
            
        Returns:
            pd.DataFrame: DataFrame map time_id bị mã hóa với true_time_id (thứ tự khôi phục).
        """
        logger.info("Đang khôi phục trật tự thời gian bằng t-SNE...")
        
        # Merge để đảm bảo có real_price cho mỗi cặp (time_id, stock_id)
        df_temp = df_train[['time_id', 'stock_id']].merge(
            df_features[['time_id', 'stock_id', 'real_price']], 
            on=['time_id', 'stock_id'], 
            how='left'
        )
        
        # Pivot: rows = time_id, columns = stock_id với giá trị là real_price
        pivot_df = df_temp.pivot(index='time_id', columns='stock_id', values='real_price')
        pivot_df = pivot_df.fillna(pivot_df.mean())
        
        # Round 1: t-SNE với perplexity cao để tìm cấu trúc tổng thể (global structure)
        perplexity_1 = min(400, pivot_df.shape[0] - 1)
        tsne_round1 = TSNE(n_components=1, perplexity=perplexity_1, 
                           random_state=self.random_state, init='pca', learning_rate='auto')
        tsne_res1 = tsne_round1.fit_transform(pivot_df.values)
        
        # Round 2: Refine bằng perplexity thấp với method exact để có độ chính xác cao nhất
        tsne_round2 = TSNE(n_components=1, perplexity=50, method='exact',
                           random_state=self.random_state, init=tsne_res1, learning_rate='auto')
        tsne_final = tsne_round2.fit_transform(pivot_df.values)
        
        # Sắp xếp để lấy true order
        time_ids = pivot_df.index.values
        order_df = pd.DataFrame({'time_id': time_ids, 'tsne_val': tsne_final.flatten()})
        order_df = order_df.sort_values('tsne_val').reset_index(drop=True)
        # Gán true_time_id theo thứ tự tăng dần
        order_df['true_time_id'] = order_df.index
        
        self.recovered_time_order = order_df[['time_id', 'true_time_id']]
        logger.info("Hoàn tất khôi phục trật tự thời gian.")
        return self.recovered_time_order

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Tạo index phân chia Train/Validation theo chiến lược Time-Series KFold nghiêm ngặt.
        Luật: Tập Train phải luôn luôn nằm ở thời điểm TRƯỚC tập Validation. Không rò rỉ dữ liệu (Zero Data Leakage).
        
        Args:
            df (pd.DataFrame): DataFrame training (BẮT BUỘC chứa cột true_time_id).
            
        Yields:
            Tuple[np.ndarray, np.ndarray]: train_idx, valid_idx (Chỉ mục numpy).
        """
        if 'true_time_id' not in df.columns:
            raise ValueError("LỖI: Cần gọi reverse_engineer_time_order và merge 'true_time_id' vào DataFrame trước khi split.")
            
        # Sắp xếp df theo time order để chuẩn bị phân chia time-series
        unique_time_ids = np.sort(df['true_time_id'].unique())
        
        # Chia tổng thời gian thành (n_splits + 1) blocks. 
        # Train tăng dần, Validation dịch lên 1 block ở mỗi fold.
        n_blocks = self.n_splits + 1
        block_size = len(unique_time_ids) // n_blocks
        
        for fold in range(self.n_splits):
            val_start = (fold + 1) * block_size
            
            # Ở fold cuối, dồn toàn bộ phần dữ liệu còn lại cho validation
            if fold == self.n_splits - 1:
                val_end = len(unique_time_ids)
            else:
                val_end = val_start + block_size
            
            # Tập train là tất cả các mốc thời gian TRƯỚC validation
            train_time_ids = unique_time_ids[:val_start]
            val_time_ids = unique_time_ids[val_start:val_end]
            
            # Map giá trị true_time_id ngược lại thành index array gốc trên df
            original_train_idx = df[df['true_time_id'].isin(train_time_ids)].index.values
            original_val_idx = df[df['true_time_id'].isin(val_time_ids)].index.values
            
            yield original_train_idx, original_val_idx
