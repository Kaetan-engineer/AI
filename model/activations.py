from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Activation(ABC):
    """
    Mathematical formulas used in neural networks to decide whether a neuron should fire or pass its signal onward.
    """
    def __init__(self) -> None:
        self.inputs = None
        self.output = None

    @abstractmethod
    def forward(self, inputs: ndarray) -> ndarray:
        """
        Runs the inputs and decides if / how it will be passed on.
        :param inputs: The inputs from the layer's pre activation.
        :return: The activated output.
        """
        return NotImplemented

    @abstractmethod
    def backward(self, output_gradient: ndarray) -> ndarray:
        """
        Runs the derivative of it's forward pass on the output gradient.
        :param output_gradient: the output gradient from the previous layer.
        :return:
        """
        return NotImplemented


class Linear(Activation):
    """
    An activation function used when no input modification is needed.
    """
    def forward(self, inputs: ndarray) -> ndarray:
        return inputs

    def backward(self, output_gradient: ndarray) -> ndarray:
        return output_gradient


class ReLU(Activation):
    """
    An activation function used (mostly in hidden layers) when negative values are unwanted in the forward pass.
    """
    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient: ndarray) -> ndarray:
        return output_gradient * (self.inputs > 0)


class Sigmoid(Activation):
    """
    An activation function used (mostly in output layers) to clip an input between 0 and 1.
    """
    def forward(self, inputs: ndarray) -> ndarray:
        self.output = np.where(
            inputs >= 0,
            1 / (1 + np.exp(-inputs)),
            np.exp(inputs) / (1 + np.exp(inputs))
        )
        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        return output_gradient * self.output * (1 - self.output)


class Tanh(Activation):
    """
    The hyperbolic tangent function used to clip the inputs between -1 and 1
    """
    def forward(self, inputs: ndarray) -> ndarray:
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        return output_gradient * (1 - self.output ** 2)


class SoftPlus(Activation):
    """
    An activation function with a smooth mathematical approximation of the ReLU activation function
    """
    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        return np.log1p(np.exp(-np.abs(inputs))) + np.maximum(inputs, 0)

    def backward(self, output_gradient: ndarray) -> ndarray:
        sigmoid = 1 / (1 + np.exp(-self.inputs))
        return output_gradient * sigmoid


class SoftMax(Activation):
    """
    An activation function used (mostly in output layers) to clip values between 0 and 1 in such a way that they all add up to 1.
    """
    def forward(self, inputs: ndarray) -> ndarray:
        shifted = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        sum_grad_output = np.sum(output_gradient * self.output, axis=-1, keepdims=True)
        return self.output * (output_gradient - sum_grad_output)