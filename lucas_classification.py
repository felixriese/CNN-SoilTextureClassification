#!/usr/bin/env python
# coding: utf-8
"""Run LUCAS classification."""

import datetime

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical

import cnn_models as cnn


def lucas_classification(data, model_name="LucasCNN", batch_size=32,
                         epochs=200, save_results=False, random_state=None,
                         verbose=0):
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
    save_results : bool (optional, default: False)
        If True, the results are saved.
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
    X_train, X_val, X_test, y_train, y_val, y_test = data
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    # 2. get model
    model = cnn.getKerasModel(model_name)

    # 3. set up callbacks
    tensorboard = TensorBoard(
        log_dir='./logs/'+run,
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

    # 7. save model
    if save_results:
        y_val_pred = model.predict(X_val, batch_size=batch_size)
        y_test_pred = model.predict(X_test, batch_size=batch_size)
        pickle.dump(y_val_pred, open("data/results/"+run+"_yvalpred.p", "wb"))
        pickle.dump(y_test_pred, open("data/results/"+run+"_ytestpred.p", "wb"))

    return score


if __name__ == '__main__':

    # CHANGE path to CSV files:
    data_path = "data/training/"

    # load data
    X_train = pd.read_csv(data_path+"X_train.csv", index_col=0).values
    X_val = pd.read_csv(data_path+"X_val.csv", index_col=0).values
    X_test = pd.read_csv(data_path+"X_test.csv", index_col=0).values
    y_train = pd.read_csv(data_path+"y_train.csv", index_col=0).values
    y_val = pd.read_csv(data_path+"y_val.csv", index_col=0).values
    y_test = pd.read_csv(data_path+"y_test.csv", index_col=0).values

    score = lucas_classification(
        data=[X_train, X_val, X_test, y_train, y_val, y_test],
        model_name="LucasCNN",
        batch_size=32,
        epochs=200,
        save_results=True,
        random_state=42)

    print(score)
