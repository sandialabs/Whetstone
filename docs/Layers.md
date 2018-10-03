# Layers


## `class whetstone.layers.Softmax_Decode(key=None, size=None, **kwargs)`

   A layer which uses a key to decode a sparse representation into a
   softmax.

   Makes it easier to train spiking classifiers by allowing the use of
   softmax and catagorical-crossentropy loss. Allows for encodings
   that are n-hot where 'n' is the number of outputs assigned to each
   class. Allows encodings to overlap, where a given output neuron can
   contribute to the probability of more than one class.

   ### Arguments
  key: A numpy array (num_classes, input_dims) with an input_dim-
  sized
 {0,1}-vector representative for each class.

  size: A tuple (num_classes, input_dim).  If `key` is not
  specified, then
 size must be specified.  In which case, a key will
 automatically be generated.

   #### build(input_shape)

  Creates the layer weights.

  Must be implemented on all layers that have weights.

  ##### Arguments
 input_shape: Keras tensor (future input to layer)
or list/tuple of Keras tensors to reference for weight
shape computations.

   #### call(inputs)

  This is where the layer's logic lives.

  ##### Arguments
 inputs: Input tensor, or list/tuple of input tensors.
 
 **kwargs: Additional keyword arguments.

  ##### Returns
 A tensor or list/tuple of tensors.

   #### compute_output_shape(input_shape)

  Computes the output shape of the layer.

  Assumes that the layer will be built to match that input shape
  provided.

  ##### Arguments
 input_shape: Shape tuple (tuple of integers)
or list of shape tuples (one per output tensor of the
layer). Shape tuples can include None for free dimensions,
instead of an integer.

  ##### Returns
 An input shape tuple.

   #### get_config()

  Returns the config of the layer.

  A layer config is a Python dictionary (serializable) containing
  the configuration of a layer. The same layer can be
  reinstantiated later (without its trained weights) from this
  configuration.

  The config of a layer does not include connectivity information,
  nor the layer class name. These are handled by *Network* (one
  layer of abstraction above).

  ##### Returns
 Python dictionary.

## `class whetstone.layers.Spiking(sharpness=0.0, **kwargs)`

   Abstract base layer for all spiking activation Layers.

   This layer should not be instantiated, but rather inherited.

   ### Arguments
  sharpness: Float, abstract 'sharpness' of the activation.
 Setting sharpness to 0.0 leaves the activation function
 unmodified. Setting sharpness to 1.0 sets the activation
 function to a threshold gate.

   #### build(input_shape)

  Creates the layer weights.

  Must be implemented on all layers that have weights.

  ##### Arguments
 input_shape: Keras tensor (future input to layer)
or list/tuple of Keras tensors to reference for weight
shape computations.

   #### get_config()

  Provides configuration info so model can be saved and loaded.

  ##### Returns
 A dictionary of the layer's configuration.

   #### sharpen(amount=0.01)

  Sharpens the activation function by the specified amount.

  ##### Arguments
 amount: Float, the amount to sharpen.

## `class whetstone.layers.Spiking_BRelu(**kwargs)`

   A Bounded Rectified Linear Unit layer that can be sharpened to a
   threshold gate.

   The sharpness value of the layer is inverted to determine the width
   of the linear-region (i.e. non-binary region), which determines the
   slope of the line in the linear-region such that the line
   intersects y = 0 and y = 1 at the current step-function borders.
   The line will always pass through the point (0.5, 0.5).

   #### build(input_shape)

  Creates the layer weights.

  Must be implemented on all layers that have weights.

  ##### Arguments
 input_shape: Keras tensor (future input to layer)
or list/tuple of Keras tensors to reference for weight
shape computations.

   #### call(inputs)

  This is where the layer's logic lives.

  ##### Arguments
 inputs: Input tensor, or list/tuple of input tensors.
 **kwargs: Additional keyword arguments.

  ##### Returns
 A tensor or list/tuple of tensors.

   #### get_config()

  Provides configuration info so model can be saved and loaded.

  ##### Returns
 A dictionary of the layer's configuration.

## `class whetstone.layers.Spiking_Sigmoid(**kwargs)`

   A Sigmoid layer that can be sharpened to a threshold gate.

   The sharpness value of the layer is inverted to determine the width
   of the linear-region (i.e. non-binary region). The roots of the
   third derivative of the sigmoid are used to map the width to a 'k'
   value which is used to scale the 'x' value in the sigmoid function,
   which places the knees of sigmoid approximately at the current
   step-function borders.

   #### build(input_shape)

  Creates the layer weights.

  Must be implemented on all layers that have weights.

  ##### Arguments
 input_shape: Keras tensor (future input to layer)
or list/tuple of Keras tensors to reference for weight
shape computations.

   #### call(inputs)

  This is where the layer's logic lives.

  ##### Arguments
 inputs: Input tensor, or list/tuple of input tensors.
 **kwargs: Additional keyword arguments.

  ##### Returns
 A tensor or list/tuple of tensors.

   #### get_config()

  Provides configuration info so model can be saved and loaded.

  ##### Returns
 A dictionary of the layer's configuration.

## `whetstone.layers.key_generator(num_classes, width, sparsity=0.1, overlapping=True)`

   Generates a key to encode and decode a one-hot vector into a sparse
   {0,1}-vector.

   ### Arguments
  num_classes: Integer, number of classes represented by the one-
  hot vector. 
  
  width: Integer, dimensionality of the expansion
  
  sparsity: Float, approximate ratio of 1's to 0's in the encoded
  vectors. overlapping: Boolean, if `False`, the encoded vectors
  are assured to

 be linearly independent.

   ### Returns
  An ndarray of size (num_classes, width)
