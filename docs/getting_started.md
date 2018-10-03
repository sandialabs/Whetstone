# Getting Started

## Overview

This package provides a framework for training deep spiking neural
networks using keras. Whetstone is designed to be extendable, modular,
and easy-to-use.

The easiest way to install is: 1. Find a location where you have write
permissions and would like a copy of the Whetstone package. 2. `git
clone` the repo This will create a new sub-directory called whetstone
which will contain all the relevant code. 3. Run `pip install .` to
install the package.


## Dependencies


- Keras

- Tensorflow (Theano or CNTK may work but are untested)

- Numpy


## Key Components


1. Whetstone is designed to be as "drop-in" as possible.  With this
   in mind, relatively few changes are required to your standard
   network.

2. *Activations*  Standard activation layers are static and
   therefore not compatible.  In `whestone.layers` you will find
   spiking-ready equivalents to standard or slightly modified
   activation functions. For example, instead of using a standard
   Rectified Linear Unit, you can use `whetstone.Spiking_BRelu` which
   is a spiking-ready version of a Bounded Rectified Linear Unit.

3. *Sharpener* You need to attach a sharpening callback.  A
   sharpener adjusts the activation functions over time, essentially
   spikifying the network.  The sharpener is responsible for
   determining when and by how much each layer should be sharpened.
   The most basic of these is `whetstone.SimpleSharperner`.

4. *Softmax_Decode* Often used for classification, this is a
   decoding layer that can be used for a softmax output layer.  The
   layer uses redundant neurons to help stabilize training.


## Example


The code below can be used to train a simple densely connected spiking
network for classifying mnist.:
```
   from __future__ import print_function
   import numpy as np
   import keras
   from keras.datasets import mnist
   from keras.models import Sequential
   from keras.utils import to_categorical
   from keras.layers import Dense
   from keras.optimizers import Adadelta
   from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
   from whetstone.callbacks import SimpleSharpener

   numClasses = 10
   (x_train, y_train),(x_test, y_test) = mnist.load_data()

   y_train = to_categorical(y_train, numClasses)
   y_test = to_categorical(y_test, numClasses)

   x_train = np.reshape(x_train, (60000,28*28))
   x_test = np.reshape(x_test, (10000,28*28))

   key = key_generator(num_classes=10, width=100)

   model = Sequential()
   model.add(Dense(256, input_shape=(28*28,)))
   model.add(Spiking_BRelu())
   model.add(Dense(64))
   model.add(Spiking_BRelu())
   model.add(Dense(100))
   model.add(Spiking_BRelu())
   model.add(Softmax_Decode(key))

   simple = SimpleSharpener(start_epoch=5, steps=5, epochs=True, bottom_up=True)

   model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=21, callbacks=[simple])

   print(model.evaluate(x_test, y_test))
```
