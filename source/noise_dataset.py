from dataclasses import dataclass
from functools import lru_cache
from math import sqrt, ceil
from random import random
from typing import Any, Callable

from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from godot.ImageTexture3D import GodotTexture3DSampler


@dataclass
class Noises:
    large:  GodotTexture3DSampler
    medium: GodotTexture3DSampler
    small:  GodotTexture3DSampler
    perlin: GodotTexture3DSampler


@dataclass
class Constants:
    large:  float
    medium: float
    small:  float
    perlin: float


@dataclass
class Data:
    noises:     Noises
    constants:  Constants
    steps:      int


class Vector3(BaseModel):
    x: float
    y: float
    z: float

    @staticmethod
    def random_direction() -> Vector3:
        return Vector3(random(), random(), random()).normalize()

    @staticmethod
    def random_start() -> Vector3:
        # [0; 1] -> [-1; 1]
        xy_coords = lambda : random() * 2.0 - 1.0

        return Vector3(xy_coords(), random(), xy_coords()).normalize()

    @staticmethod
    def dot(first: Vector3, second: Vector3) -> float:
        return first.x * second.x + first.y * second.y + first.z * second.z

    def __init__(self, x: float, y: float, z: float):
        super().__init__(x=x, y=y, z=z)
        self._length = sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def __str__(self) -> str:
        return f"{self.x} {self.y} {self.z}"

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __mul__(self, value: float) -> Vector3:
        return Vector3(
            self.x * value,
            self.y * value,
            self.z * value
        )
    
    def __truediv__(self, value: float) -> Vector3:
        return Vector3(
            self.x / value,
            self.y / value,
            self.z / value
        )
    
    def __pow__(self, value: float) -> Vector3:
        return Vector3(
            self.x ** value,
            self.y ** value,
            self.z ** value
        )

    def get(self) -> tuple[float]:
        return self.x, self.y, self.z
    
    def length(self) -> float:
        return self._length

    def normalize(self) -> Vector3:
        length: float = self.length()

        return Vector3(
            self.x / length,
            self.y / length,
            self.z / length
        )
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z], dtype=torch.float32)


def from_texture(point: Vector3, texture: GodotTexture3DSampler) -> float:
    return texture.get(*point.get())


def get_cloud_form(point: Vector3, data: Data):
    return (
        from_texture(point, data.noises.large)    * data.constants.large +
        from_texture(point, data.noises.medium)   * data.constants.medium +
        from_texture(point, data.noises.small)    * data.constants.small
    )


def get_clout_map(point: Vector3, data: Data) -> float:
    value: float = from_texture(point, data.noises.perlin)

    return (value / 2.0 + 0.5) * data.constants.perlin


def get_density(point: Vector3, data: Data) -> float:
    cloud_map: float = get_clout_map(point, data)

    value: float = get_cloud_form(point, data) * cloud_map

    if value < 0.05:
        return 0.0
    
    return value


def function(start: Vector3, end: Vector3, data: Data) -> float:
    addition: Vector3 = (end - start) / float(data.steps)
    
    position: Vector3 = start.model_copy()
    result: float = 0.0

    for _ in range(data.steps):
        local: Vector3 = position.normalize()

        result += get_density(local, data)

        position += addition
    
    return result * addition.length()


class FunctionDataset(Dataset):
    def __init__(self, size: int, params: Data, load_from: str | None = None, scale: int = 1):
        """
        scale нужен для увеличения значений случайной величины
        таким образом значения уйдут от нуля
        """

        self.size:      int = size
        self.params:    Data = params
        self.scale:     int = scale

        self.input_data = torch.empty((size, 6), dtype=torch.float32)
        self.output_data = torch.empty(size, dtype=torch.float32)

        if load_from is None:
            self.create_dataset()
            return

        self.load_dataset(load_from)

    def load_dataset(self, filepath: str):
        with open(filepath, "r") as file:
            data = file.readlines()[:self.size]
        
        assert len(data) == self.size, f"Число строк в файле меньше, чем ожидается датасетом ({len(data)} / {self.size})"

        for index in tqdm(range(self.size), desc=f"Загрузка датасета из {filepath}"):
            line = data[index].strip()

            float_data = tuple(map(float, line.split()))
            self.input_data[index] = torch.tensor(float_data[:-1], dtype=torch.float32)
            self.output_data[index] = float_data[-1] * self.scale

    def create_dataset(self):
        for index in tqdm(range(self.size), desc="Создание датасета"):
            input_1, input_2, value = self.create()
            self.input_data[index] = torch.cat([input_1.to_tensor(), input_2.to_tensor()])
            self.output_data[index] = value * self.scale
    
    @staticmethod
    def get_sphere_intersection(position: Vector3, direction: Vector3) -> Vector3:
        dot = Vector3.dot(position, direction)

        _sqrt = sqrt(dot * dot + 1.0 - position.length() * position.length())

        return position + direction * (-dot + _sqrt)

    def create(self) -> tuple[Vector3, Vector3, float]:
        start = Vector3.random_start() * 0.75
        end = self.get_sphere_intersection(start, Vector3.random_direction())
        
        value = function(start, end, self.params)

        return start, end, value

    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[index], self.output_data[index]


def dataloader_base(batch_size: int, num_workers: int) -> dict[str, Any]:
    return {
        "batch_size":   batch_size,
        "num_workers":  num_workers,
        "persistent_workers": True,
        "drop_last":          True,
    }


def get_dataset(data: Data, size: int, filepath: str | None = None, percent: float = 0.2, batch_size: int = 32, workers: int = 8) -> tuple[DataLoader, DataLoader]:
    valid_size: int = ceil(float(size) * percent)
    train_size: int = size - valid_size

    # Базовые параметры для оптимизации
    base_settings: dict[str, Any] = dataloader_base(batch_size, workers)
    
    # Каррирование для создание базового FunctionDataset по размеру
    base_dataset: Callable[[int], FunctionDataset] = lambda size: FunctionDataset(size, data, filepath)

    # Каррирование для создания DataLoader по размеру
    dataset: Callable[[bool], DataLoader] = lambda is_train: DataLoader(
        base_dataset(train_size if is_train else valid_size),
        shuffle=is_train,
        **base_settings
    )

    # Глубоким смыслом не обладает. Хотел убрать две длинные, но одинаковые строчки в возврате

    return dataset(is_train=True), dataset(is_train=False)


@lru_cache
def GET_DATA(steps: int = 1024) -> Data:
    # Функция-константа для исходных данных
    # Можете написать свою подобную

    constants = Constants(1.515, 1.375, 1.5, 0.028)
    noises = Noises(
        GodotTexture3DSampler("../dataset/LargeImage.tres"),
        GodotTexture3DSampler("../dataset/MediumImage.tres"),
        GodotTexture3DSampler("../dataset/SmallImage.tres"),
        GodotTexture3DSampler("../dataset/PerlinImage.tres")
    )

    return Data(noises, constants, steps=steps)
