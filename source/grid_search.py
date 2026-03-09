from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass

from torch.nn import Module
from torch import save


@dataclass
class SearchValue:
    model_struct:   tuple[int]
    learning_rate:  float

    def __str__(self) -> str:
        return str(self.__dict__)


class IGotModel(ABC):
    @abstractmethod
    def get_model(self) -> Module:
        """Возвращает поле с моделью"""
        pass


@dataclass
class GridSeatchParams[T: IGotModel]:
    epochs:             int
    filepath_data:      str
    filepath_model:     str
    variants:           list[SearchValue]
    convertor:          Callable[[SearchValue], T]
    train_iteration:    Callable[[T], float]
    valid_iteration:    Callable[[T], float]
    after_iteration:    Callable[[], None] | None = None


@dataclass
class SearchResult:
    scheme:         SearchValue
    model:          Module
    train_loss:     float
    valid_loss:     float


class GridSearch[T: IGotModel]:
    def __init__(self, params: GridSeatchParams[T]):
        self.best: SearchResult | None = None
        
        self.params = params
    
    def __compare(self, data: SearchResult):
        if self.best.valid_loss <= data.valid_loss:
            return
        
        self.best = data

    # Механизм автоматической остановки. 
    # Может быть только GridSeatchParams[T].epochs без улучшений
    # Иначе принудительная остановка
    def __auto_epoches(self, train_data: T) -> tuple[float, float]:
        best_train_score: float = float("+inf")
        useless_epoch_number: int = 0

        while True:
            score = self.params.train_iteration(train_data)

            if score < best_train_score:
                best_train_score = score
            elif (useless_epoch_number + 1) > self.params.epochs:
                break
            else:
                useless_epoch_number += 1
        
        validation_score: float = self.params.valid_iteration(train_data)

        return best_train_score, validation_score


    def save(self):
        with open(self.params.filepath_data, "w") as file:
             file.write(f"train: {self.best.train_loss}\nvalid: {self.best.valid_loss}\n{self.best.scheme}")

        save(self.best.model.state_dict(), self.params.filepath_model)

    def run(self):
        for variant in self.params.variants:
            data = self.params.convertor(variant)

            train, valid = self.__auto_epoches(data)

            result = SearchResult(variant, data.get_model(), train, valid)
            
            self.__compare(result)

            if self.params.after_iteration is None:
                continue
            
            self.params.after_iteration()
        
        self.save()
