import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy import io as spio
from tensorflow import keras
import pickle


def load_data(dataset_name):
    (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
        dataset_name, split=['train', 'test'], batch_size=-1, as_supervised=True,))
    x_train = np.reshape(
        x_train, [x_train.shape[0], -1]).astype('float32')/255.
    x_test = np.reshape(x_test, [x_test.shape[0], -1]).astype('float32')/255.

    return (x_train, y_train), (x_test, y_test)


def load_qmnist_data():
    mnist = pickle.load(open('./data/qmnist.pkl', "rb"))
    x_train, y_train = mnist['train_data'], mnist['train_labels']
    x_test, y_test = mnist['test_data'], mnist['test_labels']

    x_train = np.reshape(
        x_train, [x_train.shape[0], -1]).astype('float32')/255.
    x_test = np.reshape(x_test, [x_test.shape[0], -1]).astype('float32')/255.

    return (x_train, y_train), (x_test, y_test)
