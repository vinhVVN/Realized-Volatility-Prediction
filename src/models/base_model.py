from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Optional, Union

class BaseModel(ABC):
    """
    Abstract Base Class (Lớp trừu tượng) định nghĩa tiêu chuẩn thiết kế (interface) cốt lõi
    cho tất cả các mô hình học máy (Machine Learning/Deep Learning) trong dự án.
    
    Mọi model cụ thể (LightGBM, CNN, MLP, TabNet) ĐỀU PHẢI kế thừa từ class này 
    và triển khai bắt buộc 4 phương thức trừu tượng (abstract methods) bên dưới để đảm bảo 
    sự đồng nhất và nguyên tắc đa hình (polymorphism) trong thiết kế hệ thống.
    """
    
    @abstractmethod
    def train(self, 
              X_train: Union[pd.DataFrame, np.ndarray], 
              y_train: Union[pd.Series, np.ndarray],
              X_valid: Optional[Union[pd.DataFrame, np.ndarray]] = None, 
              y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
              **kwargs) -> Any:
        """
        Huấn luyện mô hình từ dữ liệu.
        
        Args:
            X_train: Tập đặc trưng huấn luyện (features).
            y_train: Nhãn mục tiêu huấn luyện (target labels).
            X_valid: Tập đặc trưng xác thực (dùng cho cơ chế Early Stopping).
            y_valid: Nhãn mục tiêu xác thực.
            **kwargs: Các siêu tham số (hyperparameters) bổ sung đặc thù của từng loại thuật toán.
            
        Returns:
            Any: Có thể trả về lịch sử huấn luyện (training history) hoặc bản thân mô hình.
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Dự đoán mục tiêu (Realized Volatility) từ tập đặc trưng đầu vào mới.
        
        Args:
            X: Tập đặc trưng đầu vào (chưa có nhãn).
            **kwargs: Các tham số cấu hình dự đoán (ví dụ: batch_size).
            
        Returns:
            np.ndarray: Mảng 1D chứa các giá trị dự đoán suy luận ra.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Lưu toàn bộ weights, trạng thái, và cấu trúc của mô hình đã huấn luyện 
        vào ổ cứng để tái sử dụng trong hệ thống production.
        
        Args:
            path (str): Đường dẫn lưu trữ (ví dụ: 'models/lgbm_fold1.pkl' hoặc 'models/cnn_fold1.pt').
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Tải weights và khôi phục trạng thái mô hình từ file lưu trữ trên ổ cứng 
        vào bộ nhớ RAM.
        
        Args:
            path (str): Đường dẫn đến file mô hình đã lưu.
        """
        pass
