from __future__ import absolute_import
from __future__ import print_function

import keras
import numpy as np
import keras.backend as K
from keras.utils import CustomObjectScope

from whetstone.layers import spikingLayersDict, Spiking


def load_model(filepath):
    """Loads a keras model that can contain custom Whetstone layers.

    Loads and returns the Keras/Whetstone model from a .h5 file at ``filepath``, handling custom layer
    deserialization for you.

    # Arguments
        filepath: Path to Keras/Whetstone model which should be a .h5 file produced by model.save(filepath).

    # Returns
        A keras Model.
    """
    with CustomObjectScope(spikingLayersDict):
        return keras.models.load_model(filepath)


def decode_from_key(key, input_vec):
    """Decodes a vector using the specified key.

    # Arguments
        key: Key used for decoding (ndarray)
        input_vec: Vector of size key.shape[1] to be decoded.

    # Returns
        Decoded one-hot vector.
    """
    #rescaled_key = np.transpose(2*key-1)
    #return [1*(np.argmax(np.matmul(2*(1-input_vec),rescaled_key))==i) for i in range(0, key.shape[0])]
    return [1*(np.argmax(np.matmul(2*key-1,2*input_vec-1))==i) for i in range(0, key.shape[0])]

def encode_with_key(key, input_vec):
    """Encodes a vector using the specified key.

    # Arguments
        key: Key used for encoding (ndarray)
        input_vec: Vector of size key.shape[0] to be encoded.

    # Returns
        Encoded {0,1}-vector.
    """
    #rescaled_key = np.transpose(2*key-1)
    #return np.matmul(rescaled_key, np.transpose(input_vec))
    #return key[np.argmax(input_vec)]
    return np.matmul(input_vec,key)


def get_spiking_layer_indices(model):
    """Returns indices of layers that can be sharpened.

    # Arguments
        model: Keras model with one or more Spiking layers.
    """
    return [i for i in range(0, len(model.layers)) if isinstance(model.layers[i], Spiking)]


def set_layer_sharpness(model, values):
    """Sets the sharpness values of all spiking layers.

    # Arguments
        model: Keras model with one or more Spiking layers.
        values: A list of sharpness values (between 0.0 and 1.0 inclusive) for each
            spiking layer in the same order as their indices.
    """
    assert type(values) is list and all([type(i) is float and i >= 0.0 and i <= 1.0 for i in values])
    for i, v in enumerate(values):
        layer = model.layers[get_spiking_layer_indices(model=model)[i]]
        K.set_value(layer.sharpness, K.cast_to_floatx(v))


def set_model_sharpness(model, value, bottom_up):
    """Sets the sharpness of the whole model.

       If ``bottom_up`` is ``True`` sharpens in bottom-up order, otherwise sharpens uniformly.

       # Arguments
            model: Keras model with one or more Spiking layers.
            value: Float, between 0.0 and 1.0 inclusive that specifies the sharpness of the model.
            bottom_up: Boolean, if ``True`` then sharpens in bottom-up order, else uniform.
    """
    assert type(value) is float and value >= 0.0 and value <= 1.0
    num_spiking_layers = len(get_spiking_layer_indices(model=model))
    if bottom_up:
        if value == 1.0: # this makes sure rounding errors don't prevent full sharpening at 1.0
            values = [1.0 for _ in range(num_spiking_layers)]
            set_layer_sharpness(model=model, values=values)
        else:
            portion_per_layer = 1.0 / num_spiking_layers
            num_fully_sharpened = int(value / portion_per_layer)
            scaled_remainder = (value % portion_per_layer) / portion_per_layer
            values = [1.0 for _ in range(num_fully_sharpened)] # for the layers already done sharpening.
            values.append(scaled_remainder) # for the layer that's currently undergoing sharpening.
            values.extend([0.0 for _ in range(num_spiking_layers - num_fully_sharpened - 1)]) # for the layers that have not yet begun to sharpen.
            set_layer_sharpness(model=model, values=values)
    else: # uniform
        values = [value for _ in range(num_spiking_layers)]
        set_layer_sharpness(model=model, values=values)
    return values


