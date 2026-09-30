from __future__ import annotations
import numpy as np
from numpy import ndarray
from model.layers import DenseLayer
from model.activations import Activation
from typing import List, Tuple, Union, Iterator
from model.loss import Loss


class NeuralNetwork:
    """
    A collection of interconnected layers used to learn patters in a given data.
    """
    def __init__(self, layer_architecture: List[Tuple[int, int, Activation]]) -> None:
        self.layers: List[DenseLayer] = list(
            DenseLayer(input_count, neuron_count, activation)
            for input_count, neuron_count, activation in layer_architecture
        )

    def weights(self) -> Iterator[ndarray]:
        """
        An iterator of all the weights in the neural network.
        :return:
        """
        for layer in self.layers:
            yield layer.weights

    def biases(self) -> Iterator[ndarray]:
        """
        An iterator of all the biases in the neural network.
        :return:
        """
        for layer in self.layers:
            yield layer.biases

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Run the inputs through the neural network and return its prediction.
        :param inputs: Inputs data for the neural network.
        :returns: The final prediction of the neural network.
        """
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output

    @staticmethod
    def loss(y_pred: ndarray, y_actual: ndarray, loss_func: Loss) -> ndarray:
        """
        The difference between the prediction and the expected results of the neural network.
        :param y_pred: The prediction that was made.
        :param y_actual: The actual / expected respose.
        :param loss_func: The loss function to be used when calculating the neural network's loss.
        :return: The calculated loss of our model.
        """
        return loss_func.forward(y_pred, y_actual)

    @staticmethod
    def loss_gradient(y_pred: ndarray, y_actual: ndarray, loss_func: Loss) -> ndarray:
        """
        The derivative of the neural network's loss.
        :param y_pred: The prediction that was made.
        :param y_actual: The actual / expected respose.
        :param loss_func: The loss function to be used when calculating the neural network's loss gradient.
        :return: The calculated loss gradient of our model.
        """
        return loss_func.backward(y_pred, y_actual)

    def backward(self, output_gradient: ndarray) -> None:
        """
        Runs the output gradient throught the neural network to update its weight and gradient derivatives.
        :param output_gradient: The loss gradient of our neural network.
        :return:
        """
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
