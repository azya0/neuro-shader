from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from model import ZmeyGorinich1
from noise_dataset import FunctionDataset, GET_DATA



@dataclass
class ProcessParams[T]:
    filepath:           str
    dataset:            torch.utils.data.Dataset
    result_function:    Callable[[T, torch.Tensor, torch.Tensor], None]
    is_skip:            Callable[[torch.Tensor, torch.Tensor], bool] | None = None


def process_model[T](params: ProcessParams):
    model = ZmeyGorinich1()

    model.load_state_dict(torch.load(params.filepath))
    model.eval()

    input_data: torch.Tensor
    output_data: torch.Tensor
    for input_data, output_data in tqdm(params.dataset, desc="Обработка данных"):
        if params.is_skip is not None and params.is_skip(input_data, output_data):
            continue
        
        with torch.no_grad():
            result: T = model(input_data.reshape(shape=(1, 6)))
        
        params.result_function(result, input_data, output_data)


class MethricCollector(ABC):
    def __init__(self, mean: float):
        self.mean:          float = mean

        self.mse:           float = 0.0
        self.variance:      float = 0.0
    
    @abstractmethod
    def collect(self, data: tuple[torch.Tensor], input_data: torch.Tensor, true_value: torch.Tensor):
        pass

    def get(self) -> float:
        return 1 - self.mse / self.variance

class R2Predictor(MethricCollector):
    def __init__(self, mean: float, index: int | None = None):
        super().__init__(mean)
        self.index = index if index is not None else 2

    def collect(self, data: tuple[torch.Tensor], input_data: torch.Tensor, true_value: torch.Tensor):
        value: float = data[self.index].item()
        
        self.mse += (value - true_value.item()) ** 2
        self.variance += (value - self.mean) ** 2


@dataclass
class ClassificatorValue:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


class F1Classificator:
    def __init__(self):
        self.value = ClassificatorValue()
    
    def __str__(self):
        precision, recall, result = self.get()

        return f"{precision=}\n{recall=}\n{result=}"

    def collect(self, data: tuple[torch.Tensor], input_data: torch.Tensor, true_value: torch.Tensor):
        # is nonzero
        nonzero_percent:        float = data[1].item()
        
        is_value_zero:       bool = true_value.item() == 0.0
        is_predicted_zero:   bool = nonzero_percent <= 0.5

        if is_predicted_zero:
            if is_value_zero:
                self.value.tp += 1
                return
            
            self.value.fp += 1
        elif is_value_zero:
            self.value.fn += 1
            return
        
        self.value.tn += 1

    def get(self) -> tuple[float]:
        # Доля истинно положительных результатов
        precision:  float = self.value.tp / (self.value.tp + self.value.fp)
        # Доля истинно положительных результатов (tp) среди всех реально положительных объектов в данных
        recall:     float = self.value.tp / (self.value.tp + self.value.fn)

        return precision, recall, 2.0 * precision * recall / (precision + recall)


class RocAucClassificator:
    def __init__(self):
        self.input: list[float] = []
        self.label: list[float] = []

    def collect(self, data: tuple[torch.Tensor], input_data: torch.Tensor, true_value: torch.Tensor):
        # input_data[1] - prob is nonzero
        self.input.append(data[1].item())
        self.label.append(0.0 if true_value.item() == 0.0 else 1.0)
    
    def show(self):
        from sklearn.metrics import roc_curve, auc

        fpr_sk, tpr_sk, _ = roc_curve(self.label, self.input)
        auc_sk = auc(fpr_sk, tpr_sk)
        print(f"AUC: {auc_sk:.3f}")

        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_sk, tpr_sk, 'g-', linewidth=2, label=f'Sklearn ROC (AUC = {auc_sk:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Случайная модель (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC-кривая (sklearn)')
        plt.legend(loc="lower right")
        
        plt.show()


class GraphCollector:
    def __init__(self):
        self.dots: list[tuple[float, float]] = []
    
    def add(self, data: tuple[torch.Tensor], input_data: torch.Tensor, true_value: torch.Tensor):
        predicted: float = data[0].item()

        self.dots.append((true_value.item(), predicted))
    
    def show(self):
        import matplotlib.pyplot as plt
        
        plt.plot(*zip(*self.dots), 'o', color='blue')
        
        import numpy as np

        x = np.linspace(0.0, 0.04, 100)

        plt.plot(x, x, color="red")

        ax = plt.gca()
        ax.set_xlabel("Истинные значения", fontsize=15)
        ax.set_ylabel("Предсказанные значения", fontsize=15)

        plt.show()

    def save_as_file(self, path: str  = "../extra_test.txt"):
        self.dots.sort(key=lambda data : data[0])

        with open(path, "w") as file:
            for x, y in self.dots:
                file.write(f"{x}\t{y}\n")


if __name__ == "__main__":
    model_path: str = "../good/second/model.pth"

    dataset = FunctionDataset(16_000, params=GET_DATA(), load_from="./dataset/data.txt")
    output_value = dataset.output_data

    output_mean: float = output_value.mean().item()
    print(f"{output_mean=}")
    nonzero_prob_mean: float = (1.0 - torch.count_nonzero(output_value) / len(output_value)).item()
    print(f"{nonzero_prob_mean=}")

    classificator_f1 = F1Classificator()
    predictor_r2 = R2Predictor(nonzero_prob_mean)

    process_model(ProcessParams(
        filepath=model_path,
        dataset=dataset,
        result_function=predictor_r2.collect,
        is_skip=lambda _input, _output: _output.item() == 0.0
    ))

    print(f"{predictor_r2.get()=}")

    process_model(ProcessParams(
        filepath=model_path,
        dataset=dataset,
        result_function=classificator_f1.collect
    ))

    print(classificator_f1)

    roc = RocAucClassificator()

    process_model(ProcessParams(
        filepath=model_path,
        dataset=dataset,
        result_function=roc.collect
    ))

    roc.show()

    all = R2Predictor(output_mean, index=0)

    process_model(ProcessParams(
        filepath=model_path,
        dataset=dataset,
        result_function=all.collect
    ))

    print(f"{all.get()=}")

    graph = GraphCollector()

    process_model(ProcessParams(
        filepath=model_path,
        dataset=dataset,
        result_function=graph.add
    ))

    graph.show()
