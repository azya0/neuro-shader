from dataclasses import dataclass

from PIL import Image
from torchinfo import summary
import torch
import typing
import threading
from queue import Queue
from functools import reduce
import subprocess

from model import MLP
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


def by_threads[T](function: typing.Callable[[int], T], full_size: int, threads_number: int = 12) -> tuple[T]:
    result: Queue[T] = Queue(maxsize=threads_number)

    threads: list[threading.Thread] = []

    part: int = full_size // threads_number
    for index in range(threads_number):
        size: int = part + (0 if index else (full_size - part * threads_number))

        thread = threading.Thread(target=lambda size: result.put(function(size)), args=(size, ))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

    return tuple([result.get() for _ in range(threads_number)])

if __name__ == "__main__":
    # image_test("../dataset/LargeImage.tres")
    # model_check("../model.pth")
    data: DataDistribution = reduce(lambda first, second : first.merge(second), by_threads(value_collector, 50000, 8))
    subprocess.run("clear")
    
    for field, value in data.__dict__.items():
        print(f"Данные для поля {field}")

        mean = reduce(lambda first, second: first + second, value) / float(len(value))
        print(f"MEAN: {mean}")
        variance = reduce(lambda first, second: first + second, [(x - mean) ** 2.0 for x in value]) / float(len(value))
        print(f"VARIANCE: {variance}")
        sigma = variance ** 0.5
        print(f"SIGMA: {sigma}")
