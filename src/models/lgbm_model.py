import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Any, Optional, Union, Dict

from src.models.base_model import BaseModel
from src.training.metrics import feval_rmspe
from src.utils.logger import get_logger

logger = get_logger("lgbm_model")

class LGBMModel(BaseModel):
    """
    Triển khai thuật toán LightGBM Gradient Boosting Framework tuân thủ BaseModel Interface.
    Được thiết kế chuyên biệt để hoạt động với hệ thống custom eval (RMSPE) của giải Optiver.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params (dict): Tập hợp các siêu tham số LightGBM (objective, learning_rate, num_leaves...).
        """
        self.params = params if params is not None else {}
        self.model = None

    def train(self, 
              X_train: Union[pd.DataFrame, np.ndarray], 
              y_train: Union[pd.Series, np.ndarray],
              X_valid: Optional[Union[pd.DataFrame, np.ndarray]] = None, 
              y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
              **kwargs) -> Any:
        
        # Thiết lập các tham số điều khiển train lgb chuẩn xác
        num_boost_round = kwargs.get('num_boost_round', 10000)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 300)
        verbose_eval = kwargs.get('verbose_eval', 100)
        categorical_feature = kwargs.get('categorical_feature', 'auto')
        weight_train = kwargs.get('weight_train', None)
        weight_valid = kwargs.get('weight_valid', None)

        # Đóng gói dữ liệu vào cấu trúc nội bộ tốc độ cao của LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight_train, categorical_feature=categorical_feature)
        
        valid_sets = [dtrain]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            # Gắn dtrain làm reference cho dvalid giúp tối ưu memory khi chia bin histogram
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, weight=weight_valid, categorical_feature=categorical_feature)
            valid_sets.append(dvalid)
            valid_names.append('valid')
            
        logger.info("Khởi động quá trình huấn luyện LightGBM...")
        
        # Tự động hóa hệ thống Callbacks
        callbacks = []
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))
            
        if X_valid is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True))

        # Train Native LightGBM API
        self.model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval_rmspe,
            callbacks=callbacks
        )
        
        logger.info(f"Hoàn tất huấn luyện. Best iteration (chống Overfitting): {self.model.best_iteration}")
        return self.model

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        if self.model is None:
            raise ValueError("LỖI: Mô hình chưa được khởi tạo weights. Hãy gọi train() hoặc load_model() trước.")
        
        # Luôn sử dụng best_iteration để suy luận nhằm bảo đảm tính tổng quát hóa
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("LỖI: Không có mô hình nào tồn tại trên bộ nhớ để lưu.")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Đã serialization mô hình xuống file: {path}")

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy file model {path}")
            
        self.model = joblib.load(path)
        logger.info(f"Đã tải thành công mô hình từ {path}")
