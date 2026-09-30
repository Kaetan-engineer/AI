import numpy as np
from tests.trainer import Trainer
from visuals.visualizer import Visualizer
from dataset.data import get_validation_data
from model.neural_network import NeuralNetwork
from model.activations import ReLU, Linear
from model.schedulers import StepLR, CosineAnnealingLR, ExponentialLR
from model.loss import MSE
from model.optimizer import Optimizer, SGD, Momentum, Adam
from typing import List

X = np.array([
    [1, 2],
    [2, 3],
    [4, 1],
    [3, 7],
    [5, 2],
    [8, 1],
], dtype=float)

y = np.array([
    [3 + 4 + 1],
    [6 + 6 + 1],
    [12 + 2 + 1],
    [9 + 14 + 1],
    [15 + 6 + 1],
    [24 + 2 + 1],
], dtype=float)

test_X = np.array([
    [10, 5],
    [7, 4],
    [2, 8],
], dtype=float)

test_y = np.array([
    [30 + 10 + 1],
    [21 + 8 + 1],
    [6 + 16 + 1]
], dtype=float)


def main():
    X_train, X_val, y_train, y_val = get_validation_data(X, y)
    model = NeuralNetwork([
        (2, 4, ReLU()),
        (4, 1, Linear())
    ])
    optimizer = Adam()
    loss = MSE()
    visualizer = Visualizer()
    scheduler = CosineAnnealingLR(
        optimizer.learning_rate,
        1e-5,
        10_000
    )
    batch_size = 6
    epochs = 10_000

    params = {
        "model": model,
        "loss_function": loss,
        "optimizer": optimizer,
        "visualizer": visualizer,
        "scheduler": scheduler,
        "batch_size": batch_size,
        "epochs": epochs
    }

    trainer = Trainer(
        **params
    )

    trainer.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        visuals=True,
        loss_record_range=100
    )

    trainer.test(
        test_X,
        test_y
    )


def compare_optimizers(*optimzers: Optimizer):
    visualizer = Visualizer()
    trainer_list: List[Trainer] = []
    loss_list: list[list[float]] = []

    for optimizer in optimzers:
        trainer_list.append(
            Trainer(
                model=NeuralNetwork([
                    (2, 4, ReLU()),
                    (4, 1, Linear())
                ]),
                loss_function=MSE(),
                optimizer=optimizer,
                visualizer=visualizer,
                epochs=10000,
                batch_size=6
            )
        )

    for trainer in trainer_list:
        loss_list.append(trainer.train(
            X,
            y,
            loss_record_range=10,
        ))

    visualizer.plot_losses()

    for trainer in trainer_list:
        prediction = trainer.model.forward(test_X)
        loss = trainer.model.loss(prediction, test_y, trainer.loss_function)

        print(f"Prediction for {str(trainer.optimizer)}:")
        print(prediction)
        print(f"Final Loss for {str(trainer.optimizer)}:")
        print(loss)
        print("\n")


def validate_test():
    X_train, X_val, y_train, y_val = get_validation_data(X, y)

    model = NeuralNetwork([
        (2, 4, ReLU()),
        (4, 1, Linear())
    ])

    optimizer = Adam()
    loss = MSE()

    trainer = Trainer(
        model=model,
        loss_function=loss,
        optimizer=optimizer,
        visualizer=Visualizer(),
        batch_size=6,
        epochs=100000
    )

    print(len(X_train))
    print(len(X_val))

    assert len(X_train) + len(X_val) == len(X)
    assert len(y_train) + len(y_val) == len(y)

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("\n")

    trainer.train(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        visuals=True,
        loss_record_range=1000
    )

if __name__ == "__main__":
    main()
