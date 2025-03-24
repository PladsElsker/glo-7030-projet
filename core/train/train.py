from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer(ABC):
    def __init__(self, model: nn.Module, parameters: "TrainingParameters") -> None:
        self.model = model
        self.parameters = parameters

    def fit(self, dataset_train: DataLoader, dataset_test: DataLoader) -> None:
        for _ in range(1, self.parameters.epochs + 1):
            for _, _ in dataset_train:
                pass

            for _, _ in dataset_test:
                pass

    @abstractmethod
    def train_one_step(self, x: tuple, y: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_one_step(self, x: tuple, y: object) -> None:
        raise NotImplementedError


@dataclass
class TrainingParameters:
    learning_rate: float
    epochs: int

    gradient_accumulation: int

    criterion: nn.Module
    optimiser: optim.Optimizer | optim.lr_scheduler.LRScheduler
