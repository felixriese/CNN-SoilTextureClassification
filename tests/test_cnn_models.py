"""Test CNN models.

Usage:
python -m pytest tests/test_cnn_models.py

"""
import os
import sys

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lucas_classification as lc


# set up dataset
X, y = make_classification(
    n_samples=100, n_features=256, n_classes=4, n_informative=8)
# X = np.expand_dims(X, 2)
# y = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
data = X_train, X_val, X_val, y_train, y_val, y_val


@pytest.mark.parametrize(
    "model_name", [
        "LucasCNN", "LucasResNet", "LucasCoordConv", "HuEtAl", "LiuEtAl",
    ])
def test_cnn_models(model_name):
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    score = lc.lucas_classification(
        data=data,
        model_name=model_name,
        batch_size=16,
        epochs=2,
        random_state=2)
    assert(isinstance(score, list))
    assert(score[1] > 0.1)
