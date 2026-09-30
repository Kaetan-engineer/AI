import numpy as np
from abc import abstractmethod, ABC
from numpy import ndarray, float64


class Loss(ABC):
    """
    A machine learning algorithm that helps us determine how bad our model did during its forward pass.
    """
    @abstractmethod
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        """
        Tells us how badly our model performed.
        :param y_pred: The prediction that was made.
        :param y_actual: The actual / expected respose.
        :return: The calculated loss of our model.
        """
        return NotImplemented

    @abstractmethod
    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        """
        The derivative of the loss' forward pass.
        :param y_pred: The prediction that was made.
        :param y_actual: The actual / expected respose.
        :return: The calculated derivative of our model.
        """
        return NotImplemented


class MSE(Loss):
    """
    A loss function that measures the average squared difference between a model's predicted values and the actual target values.
    """
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return np.mean((y_pred - y_actual) ** 2)

    def backward(self, y_pred, y_actual) -> ndarray:
        return 2 * (y_pred - y_actual) / y_pred.size


class MAE(Loss):
    """
    A loss function that measures the average magnitude of absolute errors between a model's predicted values and actual target values.
    """
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return np.mean(np.abs(y_actual - y_pred))

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        return np.sign(y_pred - y_actual) / len(y_pred)


class BCE(Loss):
    """
    A loss function that measures the performance of a classification model whose output is a probability value between 0 and 1.
    """
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return np.mean(
            (np.log(y_pred) +
             (
                     (1 - y_pred) *
                     np.log(y_actual)
             ))
        )

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        ...


class MCE(Loss):
    """
    A loss function that has a smooth, differentiable approximation of the empirical classification error rate designed for discriminative training in pattern recognition and machine learning
    """
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(y_actual * np.log(y_pred))

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -y_actual / (y_pred * y_pred.size)