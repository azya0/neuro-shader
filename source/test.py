from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
from PIL import Image
from torchinfo import summary
import typing
import threading
from queue import Queue

from model import MLP, ZmeyGorinich1
from loss_function import GorinichLoss
from noise_dataset import get_dataset, Constants, Noises, Data
from godot.ImageTextureParser import parse
from godot.ImageTexture3D import GodotTexture3DSampler
from noise_dataset import Vector3, function, GET_DATA, FunctionDataset
from tqdm import tqdm


def image_test(filename: str):
    data = parse(filename)

    image = data[0]

    Image.frombytes(mode='L', size=(
        image.width,
        image.height
    ), data = image.data.to_bytes()).show()


def value_test(filename: str):
    sample = GodotTexture3DSampler(filename)
    print(sample.get(1, 0.4, 0.1))


def dataset_test() -> float:
    constants = Constants(1.515, 1.375, 1.5, 0.028)
    noises = Noises(
        GodotTexture3DSampler("../dataset/LargeImage.tres"),
        GodotTexture3DSampler("../dataset/MediumImage.tres"),
        GodotTexture3DSampler("../dataset/SmallImage.tres"),
        GodotTexture3DSampler("../dataset/PerlinImage.tres")
    )

    data = Data(noises, constants, steps=32)

    return get_dataset(data, 10000)


def model_check(filepath: str):
    import torch

    model = MLP(sizes=(16, 16, 1), use_dropout=True)

    model.load_state_dict(torch.load(filepath))
    model.eval()

    first, second = Vector3(0.1, 0.1, 0.1), Vector3(0.2, 0.5, 0.2)
    print(f"Вектора: {first} -> {second}")

    result = function(first, second, GET_DATA())

    print(f"Результат: {result}")

    input = torch.cat([first.to_tensor(), second.to_tensor()], dim=0).unsqueeze(0)

    with torch.no_grad():
        result: torch.Tensor = model(input)

    print(f"Модель сказала: {result.item()}")


def model_test():
    data = MLP((2, 32, 64, 1))

    print(summary(data).total_params)


@dataclass
class DataDistribution:
    first:  list[Vector3]
    second: list[Vector3]
    value:  list[float]

    def merge(self, data: DataDistribution) -> DataDistribution:
        return DataDistribution(
            self.first + data.first,
            self.second + data.second,
            self.value + data.value
        )
    
    def append(self, first: Vector3, second: Vector3, value: float):
        self.first.append(first)
        self.second.append(second)
        self.value.append(value)


def value_collector(size: int = 10000) -> DataDistribution:
    dataset = FunctionDataset(size, params=GET_DATA())

    result = DataDistribution([], [], [])

    for _ in tqdm(range(len(dataset))):
        result.append(*dataset.create())

    return result


@dataclass
class SaveParams:
    filepath:   str
    file_lock:  threading.Lock


def save_dataset(size: int, params: SaveParams):
    dataset = FunctionDataset(size, params=GET_DATA())

    for _ in tqdm(range(len(dataset))):
        data = dataset.create()
        
        with params.file_lock:
            with open(params.filepath, "a") as file:
                file.write(f"{str(data[0])} {str(data[1])} {data[2]}\n")


def by_threads[T, K](function: typing.Callable[[int, K], T], params: K, full_size: int, threads_number: int = 12) -> tuple[T]:
    result: Queue[T] = Queue(maxsize=threads_number)

    threads: list[threading.Thread] = []

    part: int = full_size // threads_number
    for index in range(threads_number):
        size: int = part + (0 if index else (full_size - part * threads_number))

        thread = threading.Thread(target=lambda : result.put(function(size, params)))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

    return tuple([result.get() for _ in range(threads_number)])


def visualise_value(parts_number: int = 32):
    dataset = FunctionDataset(16_000, params=GET_DATA(), load_from="./dataset/data.txt")

    from matplotlib import pyplot as plt
    from torch import log1p

    data = dataset.output_data
    
    plt.hist(data.tolist(), bins=parts_number, color="blue")
    plt.hist(log1p(data).tolist(), bins=parts_number, color="red")

    plt.show()


def test_loss():
    import torch

    loss = GorinichLoss()
    
    test_input = (0, torch.tensor([[0.0], [0.0], [0.0]]), torch.tensor([[0.0], [0.0], [0.0]]))
    test_true = torch.tensor([[1.0], [0.0035], [0.0005]])

    print(loss(test_input, test_true))


if __name__ == "__main__":
    test_loss()
