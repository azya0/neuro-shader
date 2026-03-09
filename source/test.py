from PIL import Image
from torchinfo import summary

from model import MLP
from noise_dataset import get_dataset, Constants, Noises, Data
from godot.ImageTextureParser import parse
from godot.ImageTexture3D import GodotTexture3DSampler

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


def model_test():
    data = MLP((2, 32, 64, 1))

    print(summary(data).total_params)


if __name__ == "__main__":
    dataset_test()
