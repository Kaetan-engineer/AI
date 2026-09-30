from abc import abstractmethod, ABC
import math


class Scheduler(ABC):
    """
    An algorithm that changes a model's learning rate dynamically during training to improve convergence speed and final accuracy.
    """
    @abstractmethod
    def update(self, learning_rate: float, epochs: int) -> float:
        """
        Updates the given learning rate.
        :param learning_rate: The learning rate to update
        :param epochs: The epoch intervals to update the learning rate (only used in StepLR)
        :return:
        """
        ...


class StepLR(Scheduler):
    """
    An LR that reduces a model's learning rate by a multiplicative factor (here 0.5) at fixed epoch intervals (in this case 20).
    """
    def update(self, learning_rate: float, epoch: int) -> float:
        if epoch > 0 and epoch % 20 == 0:
            return learning_rate / 2

        return learning_rate


class ExponentialLR(Scheduler):
    """
    An LR that decreases a model's learning rate by multiplying it by a constant decay factor at each training step or epoch.
    """
    def __init__(self, decay_rate: float=0.95) -> None:
        if not 0 < decay_rate < 1:
            raise ValueError("decay_rate must be between 0 and 1")

        self.decay_rate = decay_rate

    def update(self, learning_rate: float, epochs: int) -> float:
        return learning_rate * self.decay_rate


class CosineAnnealingLR(Scheduler):
    """
    An LR that gradually reduces the learning rate during training by following the shape of a cosine function.
    """
    def __init__(self, learning_rate: float, min_learning_rate: float, total_epochs: int) -> None:
        if min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")

        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than 0")

        self.initial_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.total_epochs = total_epochs

    def update(self, learning_rate: float, epoch: int) -> float:
        cosine = math.cos(math.pi * epoch / self.total_epochs)

        return (
            self.min_learning_rate
            + 0.5
            * (self.initial_learning_rate - self.min_learning_rate)
            * (1 + cosine)
        )