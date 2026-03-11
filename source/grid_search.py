from abc import ABC, abstractmethod
from typing import Any, Callable
from dataclasses import dataclass, field

from torch.nn import Module
from torch import save


@dataclass
class SearchValue:
    learning_rate:  float
    model_args:     tuple[Any] = field(default_factory=lambda : ())
    model_kwargs:   dict[str, Any] = field(default_factory=lambda : {})
    loss_args:     tuple[Any] = field(default_factory=lambda : ())
    loss_kwargs:   dict[str, Any] = field(default_factory=lambda : {})

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
    methric_function:   Callable[[T], float]
    after_iteration:    Callable[[], None] | None = None


@dataclass
class SearchResult:
    scheme:         SearchValue
    model:          Module
    train_loss:     float
    valid_loss:     float
    methric:        float

    def __str__(self) -> str:
        ban_list: tuple[str] = ("scheme", "model")

        return str({key: value for key, value in self.__dict__.items() if not key in ban_list})


class GridSearch[T: IGotModel]:
    def __init__(self, params: GridSeatchParams[T]):
        self.best: SearchResult | None = None
        
        self.params = params
    
    def __compare(self, data: SearchResult):
        if self.best is None or self.best.methric < data.methric:
            self.best = data

    # Механизм автоматической остановки. 
    # Может быть только GridSeatchParams[T].epochs без улучшений
    # Иначе принудительная остановка
    def __auto_epoches(self, train_data: T) -> tuple[float, float]:
        best_train_score:   float = float("+inf")
        best_train_index:   int = -1

        index: int = best_train_index
        while True:
            index += 1
            score = self.params.train_iteration(train_data)

            if score < best_train_score:
                best_train_score = score
                best_train_index = index
                continue

            if (index - best_train_index) > self.params.epochs:
                break

        validation_score: float = self.params.valid_iteration(train_data)

        return best_train_score, validation_score

    def save(self):
        if self.best is None:
            print("Ни одна из моделей не доучилась до конца early-stop'а")
            return

        with open(self.params.filepath_data, "w") as file:
             file.write(f"{self.best}\n{self.best.scheme}")

        save(self.best.model.state_dict(), self.params.filepath_model)
    
    def __main(self):
        for variant in self.params.variants:
            data = self.params.convertor(variant)

            train, valid = self.__auto_epoches(data)

            methric: float = self.params.methric_function(data)

            result = SearchResult(variant, data.get_model(), train, valid, methric)
            
            self.__compare(result)

            if self.params.after_iteration is None:
                continue
            
            self.params.after_iteration()

    def run(self):
        try:
            self.__main()
        except Exception as error:
            raise error
        finally:
            self.save()
