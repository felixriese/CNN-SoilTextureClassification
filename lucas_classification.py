#!/usr/bin/env python
# coding: utf-8
"""Run LUCAS classification."""

import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import cnn_models as cnn


def lucas_classification(data, model_name="LucasCNN", batch_size=32,
                         epochs=200, random_state=None, verbose=0):
    """Run complete LUCAS classification.

    Parameters
    ----------
    data : tuple of 4 np.arrays
        Training and test data
    model_name : str (optional, default: "LUCASCNN")
        Name of the model
    batch_size : int (optional, default: 32)
        Batch size
    epochs : int (optional, default: 200)
        Number of epochs
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Returns
    -------
    score : float
        Accuracy score.

    """
    run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + model_name

    # 0. set random states
    np.random.seed(random_state)
    tf.random.set_seed(random_state+1)

    # 1. get data
    X_train, X_val, y_train, y_val = data

    # 2. get model
    model = cnn.getKerasModel(model_name)

    # 3. set up callbacks
    tensorboard = TensorBoard(
        log_dir='./logs/'+run,
        write_graph=True,
        write_grads=True,
        write_images=True,
        update_freq='epoch',
        histogram_freq=5)
    earlystopping = EarlyStopping(monitor="val_loss", patience=40)

    # 4. compile
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"])

    # 5. fit to data
    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, earlystopping])

    # 6. calculate score
    score = model.evaluate(X_val, y_val, batch_size=batch_size)
    return score


if __name__ == '__main__':

    # load data
    # (note that you first have to add the CSV files to the directory!)
    X_train = pd.read_csv("X_train.csv", index_col=0).values
    X_val = pd.read_csv("X_val.csv", index_col=0).values
    y_train = pd.read_csv("y_train.csv", index_col=0).values
    y_val = pd.read_csv("y_val.csv", index_col=0).values

    score = lucas_classification(
        data=[X_train, X_val, y_train, y_val],
        model_name="LucasCNN",
        batch_size=32,
        epochs=200,
        random_state=42)

    print(score)
