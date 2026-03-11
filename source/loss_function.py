from typing import Any
from abc import ABC, abstractmethod

from torch import nn, Tensor, tensor


class IPredictedResult(ABC):
    @abstractmethod
    def predicted_value(self, collected_data: Any) -> Tensor:
        pass


class GorinichLoss(IPredictedResult, nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()

        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
    
    def predicted_value(self, collected_data: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        return collected_data[0]

    def forward(self, data: tuple[Tensor, Tensor, Tensor], target: Tensor) -> Tensor:
        _, p_nonzero, value_pred = data

        is_zero = (target == 0).float()
        is_nonzero = 1 - is_zero
        target_nonzero = target[target > 0]
        
        classification_loss = self.bce(
            p_nonzero.squeeze(), 
            is_nonzero
        )
        
        if len(target_nonzero) > 0:
            value_pred_nonzero = value_pred[target > 0]
            regression_loss = self.mse(
                value_pred_nonzero.squeeze(), 
                target_nonzero
            )
        else:
            regression_loss: Tensor = tensor(0.0)
        
        return self.alpha * classification_loss + (1 - self.alpha) * regression_loss
