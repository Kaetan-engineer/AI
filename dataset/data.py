from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from numpy import ndarray


def get_validation_data(X: ndarray, y: ndarray) -> tuple[ndarray, ...]:
    X, y = shuffle(X, y, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)