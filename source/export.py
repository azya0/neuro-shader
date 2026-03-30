from dataclasses import dataclass

import torch

from model import ZmeyGorinich1


@dataclass
class Linear:
    name:       str
    weights:    torch.Tensor
    biases:     torch.Tensor | None


Data = Linear | tuple[str, str]


def extract_layer_matrices(model: torch.nn.Module) -> list[Data]:
    matrices: list[Data] = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.data
            biases = module.bias.data if module.bias is not None else None
            
            matrices.append(Linear(
                name=name,
                weights=weights.clone(),
                biases=biases.clone() if biases is not None else None,
            ))

            continue
        
        matrices.append((name, str(type(model))))
    
    return matrices


def save(data: list[Data], path: str):
    with open(path, "w") as file:
        write = lambda data: file.write(str(data) + "\n")

        for string in data:
            if not isinstance(string, Linear):
                write(string)
                continue
            
            write(string.name)

            weights = string.weights.tolist()

            col_size = len(weights)
            row_size = len(weights[0])
            write(f"weights ({col_size} x {row_size} = ({col_size * row_size})):")
            
            for array in weights:
                write(", ".join(map(str, array)) + ",")

            if string.biases is None:
                continue
            
            biases = string.biases.tolist()
            
            write(f"biases ({len(biases)}):")
            write(biases)


if __name__ == "__main__":
    model = ZmeyGorinich1()
    
    model.load_state_dict(torch.load("../good/second/model.pth"))
    
    save(extract_layer_matrices(model), "../export/model.wght")
