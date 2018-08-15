# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import print_function

import os, types, json, time, pickle, math, random
import shutil, errno
import keras
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.utils import Sequence
from keras.utils import CustomObjectScope
from keras.layers import InputLayer, Dense, Conv2D, Flatten, MaxPool2D, Activation

from whetstone.layers import whetstoneLayersDictionary, Spiking
from whetstone.layers import Spiking_Sigmoid, Spiking_BRelu

def load_model(filepath):
    """Loads a keras model that can contain custom Whetstone layers.

    Loads and returns the Keras/Whetstone model from a .h5 file at ``filepath``, handling custom layer
    deserialization for you.

    # Arguments
        filepath: Path to Keras/Whetstone model which should be a .h5 file produced by model.save(filepath).

    # Returns
        A keras Model.
    """
    with CustomObjectScope(whetstoneLayersDictionary):
        return keras.models.load_model(filepath)


def decode_from_key(key, input_vec):
    """Decodes a vector using the specified key.

    # Arguments
        key: Key used for decoding (ndarray)
        input_vec: Vector of size key.shape[1] to be decoded.

    # Returns
        Decoded one-hot vector.
    """
    return [1*(np.argmax(np.matmul(2*key-1,2*input_vec-1))==i) for i in range(0, key.shape[0])]


def ecncode_with_key(key, input_vec):
    """Encodes a vector using the specified key.

    # Arguments
        key: Key used for encoding (ndarray)
        input_vec: Vector of size key.shape[0] to be encoded.

    # Returns
        Encoded {0,1}-vector.
    """
    return np.matmul(input_vec,key)


def gen_schedule(num_layers, start, duration, intermission):
    """Generates a sharpening schedule for use with ScheduledSharpener.

    # Arguments
        num_layers: Integer, number of sharpenable layers in the model.
        start: Integer, epoch number on which to begin sharpening.
        duration: Integer, number of epochs over which to sharpen each layer.
        intermission: Integer, number of epochs to halt sharpening between layers.

    # Returns
        List of tuples of the form [(start_epoch, stop_epoch), (start_epoch, stop_epoch), ...]
        specifying for which epoch to to begin and end sharpening for each spiking layer, where the
        sharpening schedule for the ith spiking layer would be the ith tuple in the list.
    """
    assert type(num_layers) is int and num_layers > 0
    assert type(start) is int and start > 0
    assert type(duration) is int and duration >= 1
    assert type(intermission) is int and intermission >= 0
    current_epoch = start+duration
    schedule = [(start, current_epoch)]
    for i in range(num_layers-1):
        current_epoch += intermission
        schedule.append((current_epoch, current_epoch+duration))
        current_epoch += duration
    return schedule


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
    weights, biases = weights[0], None
    if len(weights) == 2: # (weights, biases)
        biases = weights[1]
    else: # layer was not using biases previously.
        config['use_bias'] = True # So turn biases on. We need them.
        biases = np.zeros(shape=weights.shape[-1], dtype=np.float32)
    new_layer = type(keras_preactivation_layer)(**config) # TODO There may be a more canonical way to do this.
    new_weights = weights * gamma / variance
    new_biases = (gamma / variance) * (biases - mean) + beta
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


def export_sparse(keras_sequential_model, save_path):
    """Generate a list of synapse tuples and save as a pickle file.

    Given a keras sequential model, generates a list of tuples which specify the neural connectivity
    in a general sparse format that serves as an intermediary when porting to various spiking backends.
    Each tuple specifies a source neuron, destination neuron, and connection weight. There is also a
    list giving the biases of each neuron. A layer list provides metadata about the source topology;
    specifying a list of neurons belonging to each layer and the original dimensions of the tensors.

    # Arguments
        keras_sequential_model: The Sequential model to be exported.
        save_path: path and name of pickle file to be saved.
    """
    model = copy_remove_batchnorm(keras_sequential_model=keras_sequential_model)
    # -----------------------------
    layer_metadata = []    # output tensor dimensions, neuron list.
    connectivity = []      # list of tuples of form (<presynaptic_neuron>, <postsynaptic_neuron>, <weight>)
    neuron_biases = []     # list of neuron biases
    output_encoding = {}   # output encoding metadata.
    tensors = []  # temporary list of "tensors" which are numpy arrays in the shape of the original
                  # output tensors. Each element is a neuron-id. This structure makes it easier to
                  # generate the connectivity tuples.
    next_neuron_idx = 0 # index of next neuron to be created.
    # ------------------------------
    for idx, layer in enumerate(model.layers):
        shape = [i for i in layer.output_shape if i is not None]
        n_neurons = shape[0]
        for i in shape[1:]:
            n_neurons *= i
        if type(layer) not in [Flatten, Activation, Spiking_Sigmoid, Spiking_BRelu]:
            meta = {'type':None,
                    'shape':shape,
                    'neurons':range(next_neuron_idx, next_neuron_idx + n_neurons)}
            next_neuron_idx += n_neurons
        # ------------------------------
        if type(layer) is InputLayer:
            meta['type'] = 'InputLayer'
            tensors.append(np.array(meta['neurons'], dtype=np.int).reshape(shape))
        elif type(layer) is Dense:
            meta['type'] = 'Dense'
            presynaptic_tensor = tensors[-1]
            postsynaptic_tensor = np.array(meta['neurons'], dtype=np.int).reshape(shape)
            tensors.append(postsynaptic_tensor)
            if len(presyanptic_tensor.shape) > 1:
                presynaptic_tensor = presynaptic_tensor.flatten()
            weights, biases = layer.get_weights()
            neuron_biases.extend(list(biases))
            # Determine connectivity with previous layer.
            # Each row is a presynaptic neuron, and each column is the weight
            # of the connection from said presynaptic neuron to a postsynaptic neuron.
            for pre_idx, pre_weights in weights:
                for post_idx, weight in pre_weights:
                    conn_tuple = (presynaptic_tensor[pre_idx], postsynaptic_tensor[post_idx], weight)
                    connectivity.append(conn_tuple)
        elif type(layer) is Conv2D:
            pass # TODO
        elif type(layer) is MaxPool2D:
            pass # TODO
        elif type(layer) is Activation:
            pass # ??? TODO
        elif type(layer) in [Flatten, Spiking_Sigmoid, Spiking_BRelu]:
            pass # ignore.
        else:
            return None # throw some error. TODO
        layer_metadata.append(meta)
        # Need output encoding information. TODO TODO TODO
    # ---------------------------------
    data = {'connectivity':connectivity,
            'neuron_biases':neuron_biases,
            'layer_metadata':layer_metadata,
            'output_encoding':output_encoding}
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=1)
