from torch import nn, Tensor, tensor


class GorinichLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()

        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

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


class GorinichLoss1(nn.Module):
    def __init__(self, alpha: float = 0.5, boost: float = 10_000.0):
        super().__init__()

        self.alpha = alpha
        self.boost = boost

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, data: tuple[Tensor, Tensor, Tensor], target: Tensor) -> Tensor:
        _, p_nonzero, value_pred = data

        is_zero = (target == 0.0).float()
        is_nonzero = 1.0 - is_zero
        target_nonzero = target[target > 0]

        classification_loss = self.bce(
            p_nonzero.squeeze(), 
            is_nonzero
        )

        class_loss: Tensor = self.alpha * classification_loss

        if len(target_nonzero) > 0:
            value_pred_nonzero = value_pred[target > 0]
            regression_loss = self.mse(
                value_pred_nonzero.squeeze() * self.boost, 
                target_nonzero * self.boost,
            )

            return class_loss + (1 - self.alpha) * regression_loss

        return class_loss
