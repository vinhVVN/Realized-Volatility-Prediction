import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from typing import List
from src.utils.logger import get_logger

# Bỏ qua các cảnh báo "All-NaN slice" từ numpy khi tính toán (vì có các neighbors toàn NaN)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger("nn_features")

# Kiểm tra version của pandas để sử dụng future_stack
pd_ver = pd.__version__.split('.')
SUPPORTS_FUTURE_STACK = int(pd_ver[0]) >= 2 and int(pd_ver[1]) >= 1

class NeighborFeatures:
    """
    Base class cho hệ thống Feature Engineering dựa trên K-Nearest Neighbors.
    """
    def __init__(self, metric: str = 'minkowski', p: int = 1, n_neighbors_max: int = 80):
        """
        Khởi tạo thông số thuật toán (chưa fit model tại đây).
        
        Args:
            metric (str): Phương pháp đo khoảng cách ('canberra', 'mahalanobis', 'minkowski').
            p (int): Bậc của metric Minkowski (p=1 tương đương Manhattan/L1).
            n_neighbors_max (int): Số lượng neighbors tối đa để fit.
        """
        self.metric = metric
        self.p = p
        self.n_neighbors_max = n_neighbors_max
        self.neighbor_indices = None
        self.knn = None


class TimeIdNeighbors(NeighborFeatures):
    """
    Tìm kiếm neighbors theo trục thời gian (Time-ID).
    Logic: Tìm các chu kỳ thời gian (time_id) có trạng thái thị trường (market regime) giống nhau 
    dựa trên hành vi của các cổ phiếu.
    """
    def generate_features(self, df: pd.DataFrame, pivot_col: str, target_cols: List[str], 
                          n_windows: List[int], prefix: str) -> pd.DataFrame:
        """
        Tạo NN features bằng cách nhóm theo thời gian.
        """
        logger.info(f"Đang tạo TimeIdNeighbors cho {pivot_col} với metric {self.metric}...")
        
        # 1. Pivot dữ liệu: rows = time_id, columns = stock_id
        pivot_df = df.pivot(index='time_id', columns='stock_id', values=pivot_col)
        pivot_df = pivot_df.fillna(pivot_df.mean())
        
        # Scale dữ liệu vào khoảng 0-1
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(pivot_df.values)
        
        # 2. Khởi tạo và Fit KNN model
        metric_params = None
        if self.metric == 'mahalanobis':
            # Tính pseudo-inverse (pinv) của ma trận hiệp phương sai để tránh lỗi Singular Matrix
            # rowvar=False vì features nằm ở các cột (stock_id)
            cov_matrix = np.cov(scaled_values, rowvar=False)
            VI = np.linalg.pinv(cov_matrix)
            metric_params = {'VI': VI}
            
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors_max, 
            metric=self.metric, 
            p=self.p, 
            n_jobs=-1,
            metric_params=metric_params
        )
        
        self.knn.fit(scaled_values)
        _, self.neighbor_indices = self.knn.kneighbors(scaled_values)
        
        # Khởi tạo dataframe kết quả chỉ gồm time_id và stock_id
        result_df = df[['time_id', 'stock_id']].copy()
        
        # 3. Aggregate cho từng target feature
        for target_col in target_cols:
            target_pivot = df.pivot(index='time_id', columns='stock_id', values=target_col).values
            
            for n in n_windows:
                k_indices = self.neighbor_indices[:, :n]
                neighbor_values = target_pivot[k_indices]
                
                mean_val = np.nanmean(neighbor_values, axis=1)
                min_val = np.nanmin(neighbor_values, axis=1)
                max_val = np.nanmax(neighbor_values, axis=1)
                std_val = np.nanstd(neighbor_values, axis=1)
                
                time_idx = pivot_df.index
                stock_cols = pivot_df.columns
                
                def stack_matrix(mat, feat_name):
                    temp_df = pd.DataFrame(mat, index=time_idx, columns=stock_cols)
                    if SUPPORTS_FUTURE_STACK:
                        return temp_df.stack(dropna=False, future_stack=True).reset_index(name=feat_name)
                    else:
                        return temp_df.stack(dropna=False).reset_index(name=feat_name)
                
                feat_prefix = f"{prefix}_{target_col}_{n}"
                
                df_mean = stack_matrix(mean_val, f"{feat_prefix}_mean")
                df_min = stack_matrix(min_val, f"{feat_prefix}_min")
                df_max = stack_matrix(max_val, f"{feat_prefix}_max")
                df_std = stack_matrix(std_val, f"{feat_prefix}_std")
                
                for temp_df in [df_mean, df_min, df_max, df_std]:
                    result_df = pd.merge(result_df, temp_df, on=['time_id', 'stock_id'], how='left')
                
                del k_indices, neighbor_values, mean_val, min_val, max_val, std_val
                del df_mean, df_min, df_max, df_std
                gc.collect()
            
            del target_pivot
            gc.collect()
            
        del pivot_df, scaled_values
        gc.collect()
        
        # 4. Tính toán relative rank
        base_cols = df[['time_id', 'stock_id'] + target_cols]
        result_df = pd.merge(result_df, base_cols, on=['time_id', 'stock_id'], how='left')
        
        for target_col in target_cols:
            for n in n_windows:
                feat_prefix = f"{prefix}_{target_col}_{n}"
                epsilon = 1e-8
                result_df[f"{feat_prefix}_rankmean"] = result_df[target_col] / (result_df[f"{feat_prefix}_mean"] + epsilon)
                result_df[f"{feat_prefix}_rankmax"] = result_df[target_col] / (result_df[f"{feat_prefix}_max"] + epsilon)
                result_df[f"{feat_prefix}_rankmin"] = result_df[target_col] / (result_df[f"{feat_prefix}_min"] + epsilon)
                
        result_df.drop(columns=target_cols, inplace=True)
        result_df.set_index(['time_id', 'stock_id'], inplace=True)
        return result_df


class StockIdNeighbors(NeighborFeatures):
    """
    Tìm kiếm neighbors theo trục cổ phiếu (Stock-ID).
    Logic: Gom nhóm các cổ phiếu có hành vi giá hoặc biến động giống nhau.
    """
    def generate_features(self, df: pd.DataFrame, pivot_col: str, target_cols: List[str], 
                          n_windows: List[int], prefix: str) -> pd.DataFrame:
        logger.info(f"Đang tạo StockIdNeighbors cho {pivot_col} với metric {self.metric}...")
        
        # 1. Pivot dữ liệu: rows = stock_id, columns = time_id
        pivot_df = df.pivot(index='stock_id', columns='time_id', values=pivot_col)
        pivot_df = pivot_df.fillna(pivot_df.mean())
        
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(pivot_df.values)
        
        # 2. Khởi tạo và Fit KNN
        metric_params = None
        if self.metric == 'mahalanobis':
            # Tính pseudo-inverse (pinv) cho cov matrix chống Singular Matrix
            cov_matrix = np.cov(scaled_values, rowvar=False)
            VI = np.linalg.pinv(cov_matrix)
            metric_params = {'VI': VI}
            
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors_max, 
            metric=self.metric, 
            p=self.p, 
            n_jobs=-1,
            metric_params=metric_params
        )
        
        self.knn.fit(scaled_values)
        _, self.neighbor_indices = self.knn.kneighbors(scaled_values)
        
        result_df = df[['time_id', 'stock_id']].copy()
        
        # 3. Aggregate
        for target_col in target_cols:
            target_pivot = df.pivot(index='stock_id', columns='time_id', values=target_col).values
            
            for n in n_windows:
                k_indices = self.neighbor_indices[:, :n]
                neighbor_values = target_pivot[k_indices]
                
                mean_val = np.nanmean(neighbor_values, axis=1)
                min_val = np.nanmin(neighbor_values, axis=1)
                max_val = np.nanmax(neighbor_values, axis=1)
                std_val = np.nanstd(neighbor_values, axis=1)
                
                stock_idx = pivot_df.index
                time_cols = pivot_df.columns
                
                def stack_matrix(mat, feat_name):
                    temp_df = pd.DataFrame(mat, index=stock_idx, columns=time_cols)
                    if SUPPORTS_FUTURE_STACK:
                        return temp_df.stack(dropna=False, future_stack=True).reset_index(name=feat_name)
                    else:
                        return temp_df.stack(dropna=False).reset_index(name=feat_name)
                
                feat_prefix = f"{prefix}_{target_col}_{n}"
                
                df_mean = stack_matrix(mean_val, f"{feat_prefix}_mean")
                df_min = stack_matrix(min_val, f"{feat_prefix}_min")
                df_max = stack_matrix(max_val, f"{feat_prefix}_max")
                df_std = stack_matrix(std_val, f"{feat_prefix}_std")
                
                for temp_df in [df_mean, df_min, df_max, df_std]:
                    result_df = pd.merge(result_df, temp_df, on=['time_id', 'stock_id'], how='left')
                    
                del k_indices, neighbor_values, mean_val, min_val, max_val, std_val
                del df_mean, df_min, df_max, df_std
                gc.collect()
                
            del target_pivot
            gc.collect()
            
        del pivot_df, scaled_values
        gc.collect()
        
        # 4. Tính toán relative rank
        base_cols = df[['time_id', 'stock_id'] + target_cols]
        result_df = pd.merge(result_df, base_cols, on=['time_id', 'stock_id'], how='left')
        
        for target_col in target_cols:
            for n in n_windows:
                feat_prefix = f"{prefix}_{target_col}_{n}"
                epsilon = 1e-8
                result_df[f"{feat_prefix}_rankmean"] = result_df[target_col] / (result_df[f"{feat_prefix}_mean"] + epsilon)
                result_df[f"{feat_prefix}_rankmax"] = result_df[target_col] / (result_df[f"{feat_prefix}_max"] + epsilon)
                result_df[f"{feat_prefix}_rankmin"] = result_df[target_col] / (result_df[f"{feat_prefix}_min"] + epsilon)
                
        result_df.drop(columns=target_cols, inplace=True)
        result_df.set_index(['time_id', 'stock_id'], inplace=True)
        
        return result_df
