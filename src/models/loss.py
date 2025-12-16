from torch import nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): Trọng số cho từng class (giống class_weights).
                                      Nên truyền vào biến 'class_weights' đã tính ở trên.
            gamma (float): Tham số tập trung (Focusing parameter).
                           Gamma càng lớn, model càng tập trung vào ca khó.
                           Thường chọn gamma = 2.0.
            reduction (str): 'mean' | 'sum' | 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Tính pt (xác suất dự đoán đúng)
        pt = torch.exp(-ce_loss)

        # Công thức Focal Loss: (1 - pt)^gamma * log(pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss