# Welcome to Whetstone’s documentation!
*************************************

Contents:

- [Getting Started](getting_started.md)

  - Overview

  - Dependencies

  - Key Components

  - Example

- [Examples](Examples.md)

- [Layers](Layers.md)

- [Callbacks](Callbacks.md)

- [Layer Utilities](LayerUtils.md)

- [Export Utilies](ExportUtils.md)


## Whetstone


This package provides a framework for training deep spiking neural
networks using keras. Whetstone is designed to be extendable, modular,
and easy-to-use.


## Setup


Before an initial release, you must clone the git repo and install the
package manually.

The easiest way to this is: 1. Find a location where you have write
permissions and would like a copy of the Whetstone package. 2. "git
clone https://github.com/SNL-NERL/Whetstone" This will create a new
sub-directory called whestone which will contain all the relevant
code. 3. Run "pip install ." to install the package.


## Dependencies


- Keras 2.1.5

- Tensorflow 1.3.0

- Numpy


## Example


The code below can be used to train a simple densely connected spiking
network for classifying mnist.:

```
   import numpy as np import keras from keras.datasets import mnist
   from keras.models import Sequential from keras.utils import
   to_categorical from keras.layers import Dense from whetstone.layers
   import Spiking_BRelu, Softmax_Decode from whetstone.utils import
   key_generator

   numClasses = 10 (x_train, y_train),(x_test, y_test) =
   mnist.load_data()

   y_train = to_categorical(y_train, numClasses) y_test =
   to_categorical(y_test, numClasses)

   x_train = np.reshape(x_train, (60000,28*28)) x_test =
   np.reshape(x_test, (10000,28*28))

   key = key_generator(10,100)

   model = Sequential() model.add(Dense(256, input_shape=(28*28,)))
   model.add(Spiking_BRelu()) model.add(Dense(64))
   model.add(Spiking_BRelu()) model.add(Dense(10))
   model.add(Spiking_BRelu()) model.add(Softmax_Decode(key))

   simple = SimpleSharpener(5,epochs=True)

   model.compile(loss='categorical_crossentropy', optimizer='adam') m
   odel.fit(x_train,y_train,epochs=15,callbacks=[simple],metrics=['ac
   curacy'])

   print(model.evaluate(x_test,y_test))
```
For more information, see [Getting Started](getting_started.md).
