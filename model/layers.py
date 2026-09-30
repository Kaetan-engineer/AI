from __future__ import annotations
import numpy as np
from numpy import random, ndarray
from model.activations import Activation
from model.optimizer import Optimizer


class DenseLayer:
    """
    A fully connected layer that applies a linear transformation and activation.
    """
    def __init__(self, input_count: int, neuron_count: int, activation: Activation) -> None:
        self.weights: ndarray = random.rand(neuron_count, input_count) * np.sqrt(2 / input_count)
        self.biases: ndarray = np.array(random.rand(neuron_count))

        self.activation: Activation = activation

        self.dweights = None
        self.dbiases = None
        self.dinputs = None

        self.pre_activation = None
        self.inputs = None

        self.weights_velocity = np.zeros_like(self.weights)
        self.biases_velocity = np.zeros_like(self.biases)

        self.weights_mean = np.zeros_like(self.weights)
        self.biases_mean = np.zeros_like(self.biases)

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Run the inputs through the layer and return its output.
        :param inputs: Inputs data for the layer.
        :returns: The output produced by the layer.
        """
        self.inputs = inputs

        if inputs.ndim == 1:
            self.pre_activation = self.weights @ inputs + self.biases
        else:
            self.pre_activation = inputs @ self.weights.T + self.biases

        return self.activation.forward(self.pre_activation)

    def backward(self, output_gradient: ndarray) -> ndarray:
        """
        Runs the backward pass for the layer and updates its weights / biases / inputs derivatives.
        :param output_gradient: The output gradient of the previous class.
        :return:
        """
        z_gradient = self.activation.backward(output_gradient)

        if self.inputs.ndim == 1:
            self.dweights = np.outer(z_gradient, self.inputs)
            self.dbiases = z_gradient
            self.dinputs = self.weights.T @ z_gradient

        else:
            self.dweights = z_gradient.T @ self.inputs
            self.dbiases = np.sum(z_gradient, axis=0)
            self.dinputs = z_gradient @ self.weights

        return self.dinputs

    def update(self, optimizer: Optimizer) -> None:
        """
        Updated the weights and biases of the layer.
        :param optimizer: The optimization class that should be used for updating.
        :return:
        """
        optimizer.update(self)
