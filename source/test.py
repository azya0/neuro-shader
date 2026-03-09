from PIL import Image
from torchinfo import summary
import torch

from model import MLP
from noise_dataset import get_dataset, Constants, Noises, Data
from godot.ImageTextureParser import parse
from godot.ImageTexture3D import GodotTexture3DSampler
from noise_dataset import Vector3, function, GET_DATA


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


if __name__ == "__main__":
    # image_test("../dataset/LargeImage.tres")
    model_check("../model.pth")
