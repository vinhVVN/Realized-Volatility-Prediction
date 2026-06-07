import numpy as np
from typing import Tuple

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính toán Root Mean Squared Percentage Error (RMSPE).
    Đây là metric lỗi chính thức của cuộc thi Optiver Realized Volatility Prediction.
    
    Công thức: sqrt(mean(((y_true - y_pred) / y_true) ** 2))
    
    Args:
        y_true (np.ndarray): Mảng numpy 1D chứa các nhãn thực tế (Ground truth).
        y_pred (np.ndarray): Mảng numpy 1D chứa các giá trị dự đoán (Predictions).
        
    Returns:
        float: Mức sai số RMSPE.
    """
    # Trong bài toán Optiver, target (realized volatility) không bao giờ chạm tới 0 tuyệt đối.
    # Tính toán phần trăm lỗi
    percentage_error = (y_true - y_pred) / y_true
    # Trả về căn bậc hai của trung bình bình phương phần trăm lỗi
    return np.sqrt(np.mean(np.square(percentage_error)))

def feval_rmspe(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
    """
    Custom Evaluation Metric cho framework LightGBM sử dụng RMSPE.
    
    Args:
        preds (np.ndarray): Mảng giá trị dự đoán do mô hình sinh ra.
        train_data (lgb.Dataset): Đối tượng Dataset nội bộ của LightGBM chứa nhãn thực tế.
        
    Returns:
        Tuple[str, float, bool]: 
            - Tên metric ('RMSPE')
            - Giá trị sai số
            - is_higher_better (False vì đây là hàm loss, giá trị càng thấp càng tốt)
    """
    # Lấy ground truth labels từ dataset của LightGBM
    labels = train_data.get_label()
    
    # Tính toán lỗi
    val = rmspe(labels, preds)
    
    # Trả về format chuẩn được LightGBM hỗ trợ
    return 'RMSPE', val, False
