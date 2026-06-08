import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import get_logger

logger = get_logger("nn_trainer")

def rmspe_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Hàm Loss Function RMSPE tùy chỉnh của PyTorch để ép mô hình tối ưu chuẩn xác.
    """
    loss = torch.sqrt(torch.mean(torch.square((y_true - y_pred) / (y_true + 1e-8))))
    return loss

class NNTrainer:
    """
    Trình quản lý quy trình huấn luyện PyTorch chuyên nghiệp (Orchestrator).
    Hỗ trợ tự động chạy GPU/CPU, Early Stopping, và giảm LR khi Loss đi ngang.
    """
    def __init__(self, model: nn.Module, device: str = 'auto', lr: float = 1e-3, weight_decay: float = 1e-5):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = rmspe_loss
        
        # Lịch trình giảm LR: Giảm một nửa (factor=0.5) nếu loss không giảm sau 5 epoch
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              batch_size: int = 1024, epochs: int = 100, patience: int = 15) -> float:
        """Vòng lặp huấn luyện chính."""
        
        # Chuyển đổi Numpy arrays sang PyTorch Tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_weights = None
        early_stop_counter = 0
        
        logger.info(f"Bắt đầu Training. Thiết bị: {self.device} | Epochs: {epochs} | Batch size: {batch_size}")
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            # Forward pass & Backpropagation
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                
            # Validation pass
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    preds = self.model(batch_X)
                    loss = self.criterion(preds, batch_y)
                    val_losses.append(loss.item())
                    
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # LR Scheduler nhảy bước (Dựa trên Val Loss)
            self.scheduler.step(val_loss)
            
            # Lưu Weights nếu tìm được Val Loss tốt hơn
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # Log mỗi 10 epoch hoặc khi có kỷ lục mới
            if epoch % 10 == 0 or early_stop_counter == 0:
                logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                
            # Early Stopping
            if early_stop_counter >= patience:
                logger.info(f"🛑 Early stopping kích hoạt tại epoch {epoch+1}. Dừng training.")
                break
                
        # Khôi phục trạng thái hoàn hảo nhất
        self.model.load_state_dict(best_weights)
        logger.info(f"Đã khôi phục model với Best Val RMSPE: {best_val_loss:.5f}")
        return best_val_loss
        
    def predict(self, X_test: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Thực thi suy luận (Inference)."""
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        test_loader = DataLoader(TensorDataset(X_test_t), batch_size=batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_X in test_loader:
                # Trích xuất Tensor khỏi tuple (size=1)
                pred = self.model(batch_X[0])
                preds.append(pred.cpu().numpy())
                
        return np.concatenate(preds)
        
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Đã lưu weights mô hình xuống: {path}")
        
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Đã tải thành công weights từ: {path}")
