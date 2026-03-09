from dataclasses import dataclass, field
from functools import lru_cache
import subprocess

from torch import nn, optim, Tensor, device, cuda
from torchinfo import summary
from tqdm import tqdm

from model import MLP
from noise_dataset import get_dataset, DataLoader, GET_DATA, Data
from grid_search import GridSeatchParams, SearchValue, GridSearch, IGotModel


@lru_cache
def get_device(use_cuda: bool = True) -> device:
    if not (use_cuda and cuda.is_available()):
        print("Cuda deactivated. Using cpu")
        return device("cpu")

    print(f"Cuda will use {cuda.get_device_name(0)}")

    return device("cuda")


@dataclass
class TrainData(IGotModel):
    optimizer:  optim.AdamW
    dataset:    tuple[DataLoader, DataLoader]
    model:      nn.Module
    device:     device = field(default_factory=get_device)
    loss:       nn.L1Loss = field(default_factory=nn.L1Loss)

    def get_model(self) -> nn.Module:
        return self.model


@dataclass
class TrainDataParams:
    size:       int
    percent:    float
    batch_size: int
    workers:    int
    data:       Data = field(default_factory=GET_DATA)


def create_train_data(model: nn.Module, params: TrainDataParams) -> TrainData:
    model = model.to(get_device())
    
    dataset = get_dataset(
        params.data,
        params.size,
        params.percent,
        params.batch_size,
        params.workers
    )

    optimizer = optim.AdamW(model.parameters())

    return TrainData(model=model, dataset=dataset, optimizer=optimizer)


def iteration(data: TrainData, is_train: bool) -> float:
    dataset: DataLoader = data.dataset[0 if is_train else 1]
    running_loss: float = 0.0

    data.model.train(is_train)

    input: Tensor
    output: Tensor
    for input, output in (bar := tqdm(dataset, desc=f"Starting...")):
        data.optimizer.zero_grad(set_to_none=True)

        input, output = input.to(data.device), output.to(data.device)

        result: Tensor = data.model(input)

        loss: Tensor = data.loss(result.squeeze(), output)
        loss.backward()
        data.optimizer.step()

        running_loss += loss.item()

        del input, output, result, loss

        bar.set_description(f"{"train" if is_train else "valid"} loss: {running_loss}")
    
    return running_loss


def load_dataset(structures: tuple[tuple[int]]) -> tuple[DataLoader, DataLoader]:
    print("Самая сложная модель: ")
    max_struct = max(structures, key=lambda data: sum(data))
    
    max_params = summary(MLP(max_struct, use_dropout=True)).trainable_params
    
    get_params = lambda size: TrainDataParams(
        size=size,
        percent=0.2,
        batch_size=32,
        workers=4
    )

    params = get_params(max_params * 2)

    return get_dataset(
        params.data,
        params.size,
        params.percent,
        params.batch_size,
        params.workers
    )


def main():
    if not cuda.is_available():
        print("Куда опять отказывается работать")
        return

    MLP_STRUCTURES: tuple[tuple[int]] = (
        (4, 16, 1),
        (16, 16, 1),
        (16, 32, 1),
        (32, 32, 1),
        (64, 64, 1),
    )

    
    after_iteration = lambda : subprocess.run("clear")

    dataset = load_dataset(MLP_STRUCTURES)

    after_iteration()

    LEARNING_RATES: tuple[float] = (
        0.001, 0.0001
    )

    variants: list[SearchValue] = []

    for structure in MLP_STRUCTURES:
        for lr in LEARNING_RATES:
            variants.append(
                SearchValue(structure, lr)
            )

    def convertor(data: SearchValue) -> TrainData:
        model = MLP(data.model_struct, use_dropout=True).to(get_device())

        return TrainData(
            model=model,
            optimizer=optim.AdamW(model.parameters(), lr=data.learning_rate),
            dataset=dataset,
        )
    
    params = GridSeatchParams(
        epochs=3,
        filepath_data="../best_model_data.txt",
        filepath_model="../model.pth",
        variants=variants,
        convertor=convertor,
        train_iteration=lambda data: iteration(data, is_train=True),
        valid_iteration=lambda data: iteration(data, is_train=False),
        after_iteration=after_iteration,
    )

    GridSearch(params).run()


if __name__ == "__main__":
    main()
