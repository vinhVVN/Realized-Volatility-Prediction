import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    """
    Kiến trúc 1D-Convolutional Neural Network (1D-CNN) được tối ưu cho Time-Series Data.
    Có tích hợp Residual Skip Connections (giống ResNet) để luân chuyển gradient sâu hơn.
    """
    def __init__(self, num_features: int, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Chiếu số chiều features thành hidden_size bằng một lớp Linear
        self.fc_in = nn.Linear(num_features, hidden_size)
        
        # 1D Convolutional Blocks
        # In_channels = 1 (do coi mảng hidden_size là chuỗi length), Out_channels = hidden_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(dropout)
        
        # Lớp đầu ra
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x.shape: (batch_size, num_features)
        x = self.fc_in(x)
        
        # Thêm chiều channel cho conv1d: (batch_size, 1, hidden_size)
        x = x.unsqueeze(1)
        
        # Block 1
        res1 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        # Block 2 (Có Skip Connection)
        res2 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = x + res1 # Cộng Residual
        
        # Block 3 (Có Skip Connection)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = x + res2 # Cộng Residual
        
        # Global Average Pooling (trung bình hóa trên chiều length/features)
        x = torch.mean(x, dim=2) 
        
        out = self.fc_out(x)
        
        # Trả về mảng 1D thay vì (batch_size, 1)
        return out.squeeze(-1)
