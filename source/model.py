from typing import Any

from torch import nn, Tensor


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
