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

from __future__ import print_function

"""
Light-weight demo of SimpleSharpener, Spiking_BRelu, and Softmax_Decode for a fully connected net on mnist.
"""

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger

numClasses = 10
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

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

# Create a new directory to save the logs in.
log_dir = './simple_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=21, callbacks=[simple, logger])

print(model.evaluate(x_test, y_test))
