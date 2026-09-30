from abc import abstractmethod, ABC
from typing import List
from numpy import ndarray
import numpy as np


class Optimizer(ABC):
    """
    An effective and functional class used in updating weights and biases parameters.
    """
    def __init__(self, learning_rate: float=0.01) -> None:
        self.learning_rate: float = learning_rate

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
    @abstractmethod
    def update(self, layer) -> None:
        """
        Updates the weights and biases of the given layer.
        :param layer: The layer to update.
        :return:
        """
        return NotImplemented


class SGD(Optimizer):
    """
    An optimization class used to find the model parameters that minmizes the loss function.
    """
    def __init__(self, learning_rate: float=0.01) -> None:
        super().__init__(learning_rate)

    def update(self, layer) -> None:
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


class Momentum(Optimizer):
    """
    An optimization class that speeds up gradient descent py adding a 'velocity' term to accumulate past gradients.
    """
    def __init__(self, learning_rate: float=0.01, beta: float=0.9) -> None:
        super().__init__(learning_rate)
        self.beta: float = beta

    def update(self, layer) -> None:
        layer.weights_velocity = (self.beta * layer.weights_velocity) + (1 - self.beta) * layer.dweights
        layer.biases_velocity = (self.beta * layer.biases_velocity) + (1 - self.beta) * layer.dbiases

        layer.weights -= self.learning_rate * layer.weights_velocity
        layer.biases -= self.learning_rate * layer.biases_velocity


class Adam(Optimizer):
    """
    An optimization class that merges the benefits of Momentum (which helps accelerate gradient descent and reduce oscillations) and RMSprop (which adapts the learning rate for each parameter).
    """
    def __init__(self, learning_rate: float=0.01, beta1: float=0.9, beta2: float=0.999) -> None:
        super().__init__(learning_rate)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = 1e-8
        self.t = 1

    def update(self, layer) -> None:
        layer.weights_mean = (self.beta1 * layer.weights_mean + (1 - self.beta1) * layer.dweights)
        layer.biases_mean = (self.beta1 * layer.biases_mean + (1 - self.beta1) * layer.dbiases)

        layer.weights_velocity = (self.beta2 * layer.weights_velocity + (1 - self.beta2) * layer.dweights ** 2)
        layer.biases_velocity = (self.beta2 * layer.biases_velocity + (1 - self.beta2) * layer.dbiases ** 2)

        weights_mean_corrected = layer.weights_mean / (1 - self.beta1 ** self.t)
        biases_mean_corrected = layer.biases_mean / (1 - self.beta1 ** self.t)

        weights_velocity_corrected = layer.weights_velocity / (1 - self.beta2 ** self.t)
        biases_velocity_corrected = layer.biases_velocity / (1 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * (weights_mean_corrected / (np.sqrt(weights_velocity_corrected) + self.epsilon))
        layer.biases -= self.learning_rate * (biases_mean_corrected / (np.sqrt(biases_velocity_corrected) + self.epsilon))

