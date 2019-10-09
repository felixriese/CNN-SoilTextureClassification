"""Test CNN models.

Usage:
python -m pytest tests/test_cnn_models.py

"""
import sys
import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lucas_classification as lc

# set up dataset
X, y = make_classification(
    n_samples=100, n_features=256, n_classes=4, n_informative=8)
X = np.expand_dims(X, 2)
y = to_categorical(y, 4)
data = train_test_split(X, y, test_size=0.5)


@pytest.mark.parametrize(
    "model_name", [
        "LucasCNN", "LucasResNet", "LucasCoordConv", "HuEtAl", "LiuEtAl",
    ])
def test_cnn_models(model_name):
    score = lc.lucas_classification(
        data=data,
        model_name=model_name,
        batch_size=16,
        epochs=2,
        random_state=2)
    assert(isinstance(score, list))
    assert(score[1] > 0.1)
