import matplotlib.pyplot as plt
from numpy import ndarray


class Visualizer:
    """
    A class used to plot the different performaces of a model with matplotlib.
    """
    def __init__(self) -> None:
        self.losses = {}

    def record_loss(self, optimizer_name: str,  epoch: int, training_loss: float | ndarray, validation_loss: float| ndarray) -> None:
        """
        Saves the given parameters inside its object.
        :param optimizer_name: The name of the optimization used.
        :param epoch: The current epoch.
        :param training_loss: The training loss of the model.
        :param validation_loss: The classification loss of the model.
        :return:
        """
        if optimizer_name not in self.losses:
            self.losses[optimizer_name] = {
                "Epochs": [],
                "Training Losses": [],
                "Validation Losses": []
            }

        self.losses[optimizer_name]["Epochs"].append(epoch)
        self.losses[optimizer_name]["Training Losses"].append(training_loss)
        self.losses[optimizer_name]["Validation Losses"].append(validation_loss)

    def plot_losses(self) -> None:
        """
        Plots the recorded losses on a matplotlib plot.
        :return:
        """
        for optimizer_name, data in self.losses.items():
            plt.plot(
                data["Epochs"],
                data["Training Losses"],
                label=f"{optimizer_name} - Training"
            )

            plt.plot(
                data["Epochs"],
                data["Validation Losses"],
                label=f"{optimizer_name} - Validation"
            )

        plt.title("Neural Network Loss Visualization")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.grid()
        plt.show()