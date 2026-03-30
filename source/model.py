from typing import Any

from torch import nn, Tensor, sigmoid, relu


class MLP(nn.Module):
    def __init__(self, sizes: tuple[float], use_dropout: bool = False):
        super().__init__()

        assert len(sizes) >= 1
        
        sequence: list[Any] = []

        for index, size in enumerate(sizes):
            from_size = 6 if index == 0 else sizes[index - 1]

            sequence.append(
                nn.Linear(from_size, size)
            )

            if not index:
                sequence.append(
                    nn.BatchNorm1d(size)
                )

            if not use_dropout:
                continue

            sequence.append(
                nn.Dropout(p=0.1)
            )
        
        sequence.append(nn.ReLU())
        
        self.sequence = nn.Sequential(*sequence)
        

    def forward(self, x: Tensor) -> float:
        return self.sequence(x)


class ZmeyGorinich1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(32, 1)
        
        self.regressor = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, data: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.shared(data)

        p_nonzero = sigmoid(self.classifier(features))

        value_pred = self.regressor(features)
        value_pred = relu(value_pred)
        
        final_pred = p_nonzero * value_pred
        
        return final_pred, p_nonzero, value_pred


class ZmeyGorinich2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.shared_classifier = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(32, 1)
        
        self.shared_regressor = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features_classifier = self.shared_classifier(data)

        p_nonzero = sigmoid(self.classifier(features_classifier))

        features_regressor = self.shared_regressor(data)

        value_pred = self.regressor(features_regressor)
        value_pred = relu(value_pred)
        
        final_pred = p_nonzero * value_pred
        
        return final_pred, p_nonzero, value_pred
