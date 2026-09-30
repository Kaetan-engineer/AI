import numpy as np
from numpy import ndarray, floating, float64
from model.neural_network import NeuralNetwork
from model.loss import Loss
from model.schedulers import Scheduler
from sklearn.utils import shuffle
from model.optimizer import Optimizer, Adam
from visuals.visualizer import Visualizer
from typing import Any


class Trainer:
    """
    A class used to supervises the training and testing of a model against a set of data.
    """
    def __init__(self, model: NeuralNetwork, loss_function: Loss, optimizer: Optimizer, visualizer: Visualizer, scheduler: Scheduler, batch_size: int, epochs: int) -> None:
        self.model: NeuralNetwork = model
        self.loss_function: Loss = loss_function
        self.optimizer: Optimizer = optimizer
        self.visualizer: Visualizer = visualizer
        self.scheduler: Scheduler = scheduler
        self.batch_size: int = batch_size
        self.epochs: int = epochs

    def train(self,
              X: ndarray,
              y: ndarray,
              X_val: ndarray,
              y_val: ndarray,
              visuals: bool=False,
              loss_record_range: int=10,
              patience_limit: int=5
        ) -> list[float64]:
        """
        Trains the model with the given set of data.
        :param X: The data to train with.
        :param y: The expected result of the training data.
        :param X_val: The data to validate the model's learning pattern.
        :param y_val: The expected result of the validation data.
        :param visuals: Determines whether you want printing statements and a matplotlib plot.
        :param loss_record_range: Determines the epoch intervals of when the loss will be registered.
        :param patience_limit: The maximum amount of times the model can do worse that its best loss performance.
        :return: The training losses of the model.
        """
        loss_history = {
            "Training": [],
            "Validation": []
        }
        best_model = {}
        patience_counter = 0
        X_train, y_train = X.copy(), y.copy()

        for epoch in range(self.epochs):
            X_shuffled, y_shuffled = shuffle(X_train, y_train)

            for start in range(0, len(X_shuffled), self.batch_size):
                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                prediction = self.model.forward(X_batch)
                loss = self.model.loss(prediction, y_batch, self.loss_function)

                loss_gradient = self.model.loss_gradient(
                    prediction,
                    y_batch,
                    self.loss_function
                )

                self.model.backward(loss_gradient)

                for layer in self.model.layers:
                    self.optimizer.update(layer)

                if isinstance(self.optimizer, Adam):
                    self.optimizer.t += 1

                self.optimizer.learning_rate = self.scheduler.update(self.optimizer.learning_rate, epoch)

            if (epoch + 1) % loss_record_range == 0:
                weights_before = [
                    layer.weights.copy()
                    for layer in self.model.layers
                ]

                training_loss = self.validate(X_train, y_train)
                validation_loss = self.validate(X_val, y_val)

                loss_history["Training"].append(training_loss)
                loss_history["Validation"].append(validation_loss)

                validation_loss = loss_history["Validation"][-1]
                best_validation_loss = np.inf
                min_delta = 1e-6

                if validation_loss < best_validation_loss - min_delta:
                    best_validation_loss = validation_loss
                    patience_counter = 0
                    best_model["Weights"] = [weights.copy() for weights in self.model.weights()]
                    best_model["Biases"] = [biases.copy() for biases in self.model.biases()]
                else:
                    patience_counter += 1

                if patience_counter >= patience_limit:
                    print("Patience limit exceeded")
                    break

                self.visualizer.record_loss(
                    str(self.optimizer),
                    epoch + 1,
                    training_loss,
                    validation_loss
                )

                if visuals:
                    print(f"Epoch: {epoch + 1}")
                    print(f"Training Loss: {training_loss:.10f}")
                    print(f"Validation Loss: {validation_loss:.10f}")
                    print(f"Learning rate: {self.optimizer.learning_rate}")
                    print("max |dW|:", max(np.max(np.abs(layer.dweights)) for layer in self.model.layers))
                    print("max |db|:", max(np.max(np.abs(layer.dbiases)) for layer in self.model.layers))

                    print("\n")

        for layer, weights, biases in zip(
                self.model.layers,
                best_model["Weights"],
                best_model["Biases"]
        ):
            layer.weights = weights.copy()
            layer.biases = biases.copy()

        if visuals:
            self.visualizer.plot_losses()

        return loss_history["Training"]

    def validate(self, X: ndarray, y: ndarray) -> ndarray:
        """
        Validates the given data without any backward pass.
        :param X: The data to validate the model's learning pattern.
        :param y: The expected result of the validation data.
        :return: The loss of the given data.
        """
        prediction = self.model.forward(X)

        return self.model.loss(prediction, y, self.loss_function)

    def test(self, *args: ndarray) -> None:
        """
        Tests the model against unsees data.
        :param args: All the data + expected result to test with.
        :return:
        """
        prediction = self.model.forward(args[0])
        print(f"Prediction: {prediction}")
        if len(args) > 1:
            print(f"Accuracy: {float(self.accuracy(args[0], args[1]))}")

    @staticmethod
    def accuracy(prediction: ndarray, actual: ndarray) -> floating[Any]:
        """
        Measures the accuracy of the model compared to the wanted result.
        :param prediction: The model's prediction.
        :param actual: The expected results.
        :return: The calculated accuracy.
        """
        prediction, actual = np.argmax(prediction, axis=1), np.argmax(actual, axis=1)
        return np.mean(prediction == actual) * 100