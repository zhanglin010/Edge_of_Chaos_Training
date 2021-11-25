import os
import pickle
from collections import OrderedDict

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
import tensorflow as tf
from tensorflow import keras


class LoggingReporter(keras.callbacks.Callback):
    """Save the activations to files at after some epochs.

    Args:
        args: configuration options dictionary
        x_test: test data
        y_test: test label
    """

    def __init__(self, args, x_train, y_train, x_test, y_test, *kargs, **kwargs):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.args = args
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs=None):
        if not os.path.exists(self.args.save_weights_dir):
            os.makedirs(self.args.save_weights_dir)
        if not os.path.exists(self.args.save_losses_dir):
            os.makedirs(self.args.save_losses_dir)
        if not os.path.exists(self.args.save_scores_dir):
            os.makedirs(self.args.save_scores_dir)
        self.losses = {}
        self.losses['train'] = []
        self.losses['val'] = []
        self.W0_init = self.model.get_weights()[0]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.args.log_epochs:
            # Compute weights
            weights = save_weights(self.args, self.W0_init, self.model)
            # Save the weights
            fname = '{}/epoch{:05d}'.format(self.args.save_weights_dir, epoch)
            with open(fname, 'wb') as f:
                pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
        if epoch == 0:
            self.losses['train'].append(self.model.evaluate(self.x_train,
                                                            self.y_train,
                                                            verbose=0))
            self.losses['val'].append(self.model.evaluate(self.x_test,
                                                          self.y_test,
                                                          verbose=0))

    def on_epoch_end(self, epoch, logs=None):
        # print(logs.keys())
        self.losses['train'].append(
            [logs['loss'], logs['sparse_categorical_crossentropy'], logs['accuracy']])
        self.losses['val'].append(
            [logs['val_loss'], logs['val_sparse_categorical_crossentropy'], logs['val_accuracy']])
        # self.losses['train'].append(logs)
        # self.losses['val'].append(logs)

    def on_train_end(self, logs=None):
        # save training losses to file
        fname = '{}/losses'.format(self.args.save_losses_dir)
        with open(fname, 'wb') as f:
            pickle.dump(self.losses, f, pickle.HIGHEST_PROTOCOL)

        # save scores to file
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        fname = '{}/scores'.format(self.args.save_scores_dir)
        with open(fname, 'wb') as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)


def save_weights(args, W0_init, model):
    results = {}
    # W0 is 1st layer weights matrix
    W0 = model.get_weights()[0]
    W_diff = np.linalg.norm(W0-W0_init)
    # print('shape of difference is {}'.format((W0-W0_init).shape))
    results['W0s'] = W0
    results['W_diffs'] = W_diff

    return results
