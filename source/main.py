from dataclasses import dataclass, field
from functools import lru_cache
import subprocess

from torch import nn, optim, Tensor, device, cuda, no_grad
from tqdm import tqdm

from model import MLP, ZmeyGorinich1, ZmeyGorinich2
from loss_function import GorinichLoss1
from noise_dataset import FunctionDataset, get_dataset, DataLoader, GET_DATA, Data
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
    loss:       nn.Module
    model:      nn.Module = field(default_factory=ZmeyGorinich2)
    device:     device = field(default_factory=get_device)

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

        result = data.model(input)

        loss: Tensor = data.loss(result, output)
        loss.backward()
        data.optimizer.step()

        running_loss += loss.item() / len(dataset)

        del input, output, result, loss

        bar.set_description(f"{"train" if is_train else "valid"} loss: {running_loss}")
    
    return running_loss


def r2_metric(data: TrainData) -> float:
    data.model.eval()
    dataset: FunctionDataset = data.dataset[1].dataset

    mean = dataset.output_data.mean().item()

    mse: float = 0.0
    variance: float = 0.0

    input_value: Tensor
    output_value: Tensor
    for input_value, output_value in tqdm(dataset, desc="Оцениваем модель..."):
        with no_grad():
            result: tuple[Tensor] = data.model(input_value.to(get_device()))

        predicted: Tensor = result[0]

        mse += (output_value.item() - predicted.item()) ** 2
        variance += (mean - predicted.item()) ** 2

        del input_value, output_value, predicted
        
        for tensor in result:
            del tensor
        

    return 1 - mse / variance


def load_dataset(size: int, filepath: str | None = None) -> tuple[DataLoader, DataLoader]:
    params = TrainDataParams(
        size=size,
        percent=0.2,
        batch_size=64,
        workers=8
    )

    return get_dataset(
        params.data,
        params.size,
        filepath,
        params.percent,
        params.batch_size,
        params.workers
    )


def main():
    if not cuda.is_available():
        print("Куда опять отказывается работать")
        return
    
    after_iteration = lambda : subprocess.run("clear")

    dataset = load_dataset(16_000, "./dataset/data.txt")

    after_iteration()

    LOSS_KWARGS: tuple[tuple[int]] = (
        {"alpha": 0.5, "boost": 10.0},
        {"alpha": 0.5, "boost": 100.0},
        {"alpha": 0.5, "boost": 500.0},
        {"alpha": 0.5, "boost": 1_000.0},
        {"alpha": 0.5, "boost": 5_000.0},
        {"alpha": 0.5, "boost": 10_000.0},
    )

    LEARNING_RATES: tuple[float] = (
        0.001, 0.0001,
    )

    variants: list[SearchValue] = []

    for lr in LEARNING_RATES:
        for loss_kwarg in LOSS_KWARGS:
            variants.append(
                SearchValue(lr, loss_kwargs=loss_kwarg)
            )

    def convertor(data: SearchValue) -> TrainData:
        model = ZmeyGorinich2(*data.model_args, **data.model_kwargs).to(get_device())
        
        loss = GorinichLoss1(*data.loss_args, **data.loss_kwargs)

        return TrainData(
            model=model,
            optimizer=optim.AdamW(model.parameters(), lr=data.learning_rate),
            dataset=dataset,
            loss=loss,
        )
    
    params = GridSeatchParams(
        epochs=15,
        filepath_data="../best_model_data.txt",
        filepath_model="../model.pth",
        variants=variants,
        convertor=convertor,
        train_iteration=lambda data: iteration(data, is_train=True),
        valid_iteration=lambda data: iteration(data, is_train=False),
        metric_function=r2_metric,
        after_iteration=after_iteration,
    )

    GridSearch(params).run()


if __name__ == "__main__":
    main()
