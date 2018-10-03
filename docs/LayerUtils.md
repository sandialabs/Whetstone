# LayerUtils


## `whetstone.utils.layer_utils.decode_from_key(key, input_vec)`

   Decodes a vector using the specified key.

   ### Arguments
  key: Key used for decoding (ndarray) 
  
  input_vec: Vector of size key.shape[1] to be decoded.

   ### Returns
  Decoded one-hot vector.

## `whetstone.utils.layer_utils.encode_with_key(key, input_vec)`

   Encodes a vector using the specified key.

   ### Arguments
  key: Key used for encoding (ndarray) 
  input_vec: Vector of size key.shape[0] to be encoded.

   ### Returns
  Encoded {0,1}-vector.

## `whetstone.utils.layer_utils.get_spiking_layer_indices(model)`

   Returns indices of layers that can be sharpened.

   ### Arguments
  model: Keras model with one or more Spiking layers.

## `whetstone.utils.layer_utils.load_model(filepath)`

   Loads a keras model that can contain custom Whetstone layers.

   Loads and returns the Keras/Whetstone model from a .h5 file at
   `filepath`, handling custom layer deserialization for you.

   ### Arguments
  filepath: Path to Keras/Whetstone model which should be a .h5
  file produced by `model.save(filepath)`.

   ### Returns
  A keras Model.

## `whetstone.utils.layer_utils.set_layer_sharpness(model, values)`

   Sets the sharpness values of all spiking layers.

   ### Arguments
  model: Keras model with one or more Spiking layers. 
  
  values: A list of sharpness values (between 0.0 and 1.0 inclusive) for
  each

 spiking layer in the same order as their indices.

## `whetstone.utils.layer_utils.set_model_sharpness(model, value, bottom_up)`

   Sets the sharpness of the whole model.

   If `bottom_up` is `True` sharpens in bottom-up order, otherwise
   sharpens uniformly.

   ### Arguments
  model: Keras model with one or more Spiking layers. 
  
  value: Float, between 0.0 and 1.0 inclusive that specifies the
  sharpness of the model. 
  
  bottom_up: Boolean, if `True` then sharpens in bottom-up order, else uniform.
