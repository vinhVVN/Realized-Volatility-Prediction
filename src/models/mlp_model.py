import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    Kiến trúc Multi-Layer Perceptron (Mạng nơ-ron truyền thẳng đa lớp) sử dụng hàm kích hoạt SiLU.
    Có trang bị BatchNorm và Dropout chống Overfitting.
    """
    def __init__(self, num_features: int, hidden_sizes: list = [512, 256, 128, 64], dropout: float = 0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_dim = num_features
        
        for h in hidden_sizes:
            block = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.SiLU(), # SiLU thường hội tụ tốt hơn ReLU trong Deep Learning tài chính
                nn.Dropout(dropout)
            )
            self.layers.append(block)
            in_dim = h
            
        self.fc_out = nn.Linear(in_dim, 1)
        
    def forward(self, x):
        # Truyền tuần tự qua các layers
        for idx, layer in enumerate(self.layers):
            # Nếu 2 layer liên tiếp có cùng số out_features (không phổ biến trong MLP giảm dần),
            # ta có thể thêm residual block. Nếu không thì cứ truyền thẳng.
            x = layer(x)
                
        out = self.fc_out(x)
        return out.squeeze(-1)
