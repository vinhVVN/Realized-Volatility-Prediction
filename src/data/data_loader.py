import os
import glob
from enum import Enum
import pandas as pd
from joblib import Parallel, delayed
from src.utils.logger import get_logger

logger = get_logger("data_loader")

class DataBlock(Enum):
    """Enum để xác định khối dữ liệu cần load."""
    TRAIN = 'train'
    TEST = 'test'
    BOTH = 'both'

class DataLoader:
    """
    Class hỗ trợ đọc dữ liệu Parquet đa luồng từ thư mục dự án Optiver.
    """
    def __init__(self, data_dir: str):
        """
        Khởi tạo DataLoader.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu raw.
        """
        self.data_dir = data_dir
        
    def _read_train_test(self, block: DataBlock = DataBlock.TRAIN) -> pd.DataFrame:
        """
        Đọc file train.csv hoặc test.csv.
        """
        if block == DataBlock.TRAIN:
            path = os.path.join(self.data_dir, "train.csv")
            if os.path.exists(path):
                return pd.read_csv(path)
            else:
                logger.warning(f"File không tồn tại: {path}")
                return pd.DataFrame()
        elif block == DataBlock.TEST:
            path = os.path.join(self.data_dir, "test.csv")
            if os.path.exists(path):
                return pd.read_csv(path)
            else:
                logger.warning(f"File không tồn tại: {path}")
                return pd.DataFrame()
        elif block == DataBlock.BOTH:
            train_df = self._read_train_test(DataBlock.TRAIN)
            test_df = self._read_train_test(DataBlock.TEST)
            return pd.concat([train_df, test_df], ignore_index=True)
        else:
            raise ValueError(f"Block không hợp lệ: {block}")
            
    def get_target_dataframe(self, block: DataBlock = DataBlock.TRAIN) -> pd.DataFrame:
        """Lấy dataframe chứa nhãn mục tiêu và các cột liên quan."""
        return self._read_train_test(block)
        
    @staticmethod
    def _read_single_stock(file_path: str) -> pd.DataFrame:
        """Hàm đọc 1 file parquet đơn lẻ (dành cho Parallel)."""
        # Đường dẫn thư mục Parquet chia theo stock_id (vd: .../stock_id=123/xxxx.parquet)
        try:
            # Xử lý separator linh hoạt cho cả Windows và Linux
            normalized_path = file_path.replace("\\", "/")
            stock_id = int(normalized_path.split("stock_id=")[1].split("/")[0])
            
            df = pd.read_parquet(file_path)
            df['stock_id'] = stock_id
            return df
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {file_path}: {e}")
            return pd.DataFrame()

    def load_parquet_parallel(self, data_type: str, block: DataBlock = DataBlock.TRAIN, n_jobs: int = -1) -> pd.DataFrame:
        """
        Đọc tất cả các file parquet của một loại dữ liệu ('book' hoặc 'trade') song song.
        
        Args:
            data_type (str): 'book' hoặc 'trade'.
            block (DataBlock): Nhóm dữ liệu cần lấy (TRAIN hoặc TEST).
            n_jobs (int): Số lượng core sử dụng để xử lý song song (-1: dùng tất cả).
            
        Returns:
            pd.DataFrame: DataFrame tổng hợp chứa tất cả dữ liệu.
        """
        if data_type not in ['book', 'trade']:
            raise ValueError("data_type phải là 'book' hoặc 'trade'")
            
        if block == DataBlock.BOTH:
            df_train = self.load_parquet_parallel(data_type, DataBlock.TRAIN, n_jobs)
            df_test = self.load_parquet_parallel(data_type, DataBlock.TEST, n_jobs)
            return pd.concat([df_train, df_test], ignore_index=True)
            
        target_dir = os.path.join(self.data_dir, f"{data_type}_{block.value}.parquet")
        
        if not os.path.exists(target_dir):
            logger.warning(f"Thư mục không tồn tại: {target_dir}")
            return pd.DataFrame()
            
        # Tìm tất cả các file parquet con
        file_patterns = os.path.join(target_dir, "stock_id=*", "*.parquet")
        files = glob.glob(file_patterns)
        
        if not files:
            logger.warning(f"Không tìm thấy file parquet nào trong: {target_dir}")
            return pd.DataFrame()
            
        logger.info(f"Đang đọc song song {len(files)} file {data_type} {block.value}...")
        
        dfs = Parallel(n_jobs=n_jobs)(delayed(DataLoader._read_single_stock)(file) for file in files)
        
        # Loại bỏ các dataframe rỗng nếu có lỗi
        dfs = [df for df in dfs if not df.empty]
        
        if dfs:
            result_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Hoàn tất đọc dữ liệu. Kích thước tổng: {result_df.shape}")
            return result_df
        else:
            return pd.DataFrame()
