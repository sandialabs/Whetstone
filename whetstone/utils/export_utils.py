from __future__ import absolute_import
from __future__ import print_function

import os, pickle, copy
import keras
import numpy as np
import keras.backend as K
from keras.layers import InputLayer, Dense, Conv1D, Conv2D, Flatten, MaxPool2D, Activation

from whetstone.layers import Spiking_BRelu, Spiking_Sigmoid, Softmax_Decode
from whetstone.utils.layer_utils import load_model

def merge_batchnorm(keras_preactivation_layer, keras_batchnorm_layer):
    """Merges the parameters of a batch normalization layer into the layer before it.

    # Arguments
        keras_preactivation_layer: Layer directly preceding the batchnorm layer.
        keras_batchnorm_layer: The batch normalization layer to be merged into the preactivation layer.

    # Returns
        new_layer, (new_weights, new_biases) : Where new_layer is a keras layer of the same configuration
        as keras_preactivation_layer, and (new_weights, new_biases) are the updated weights and biases
        for the new_layer to be used with new_layer.set_weights(((new_weights, new_biases))) after the
        new model is built.
    """
    assert type(keras_batchnorm_layer) == keras.layers.normalization.BatchNormalization
    (gamma, beta, mean, variance) = keras_batchnorm_layer.get_weights()
    config = keras_preactivation_layer.get_config()
    weights = keras_preactivation_layer.get_weights()
    if len(weights) == 2: # (weights, biases)
        weights, biases = weights[0], weights[1]
    else: # layer was not using biases previously.
        weights = weights[0]
        config['use_bias'] = True # So turn biases on. We need them.
        biases = np.zeros(shape=weights.shape[-1], dtype=np.float32)
    new_layer = type(keras_preactivation_layer)(**config) # TODO There may be a more canonical way to do this.
    stdev = np.sqrt(variance)
    new_weights = weights * gamma / (stdev + K.epsilon())
    new_biases = (gamma / (stdev + K.epsilon())) * (biases - mean) + beta
    return new_layer, (new_weights, new_biases)


def copy_remove_batchnorm(keras_sequential_model):
    """Make a functionally equivalent copy of a net with the BatchNormalization layers removed.

    Given a keras Sequential model, returns an equivalent Sequential model without batchnorm layers.
    Assumes you only use batch normalization directly after a layer with activation=None.

    # Arguments
        keras_sequential_model: The Sequential model to be copied.

    # Returns
        new_model: A copy of keras_sequential_model with the batch normalization layers removed.
    """
    new_model = keras.models.Sequential()
    layers = keras_sequential_model.layers
    dict_new_weights = {}
    idx = 0
    while idx < len(layers):
        layer = layers[idx]
        next_layer = layers[idx+1] if idx < len(layers)-1 else None
        if type(next_layer) == keras.layers.normalization.BatchNormalization:
            new_layer, new_weights = merge_batchnorm(layer, next_layer)
            dict_new_weights[new_layer.name] = new_weights
            new_model.add(new_layer)
            idx += 2
        else:
            new_model.add(type(layer)(**layer.get_config()))
            dict_new_weights[layer.name] = layer.get_weights()
            idx += 1
    new_model.build()
    for layer in new_model.layers:
        if layer.name in dict_new_weights:
            layer.set_weights(dict_new_weights[layer.name])
    return new_model

