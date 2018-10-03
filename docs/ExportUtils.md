# ExportUtils


## `whetstone.utils.export_utils.copy_remove_batchnorm(keras_sequential_model)`

   Make a functionally equivalent copy of a net with the
   BatchNormalization layers removed.

   Given a keras Sequential model, returns an equivalent Sequential
   model without batchnorm layers. Assumes you only use batch
   normalization directly after a layer with activation=None.

   ### Arguments
  keras_sequential_model: The Sequential model to be copied.

   ### Returns
  new_model: A copy of keras_sequential_model with the batch
  normalization layers removed.

## `whetstone.utils.export_utils.merge_batchnorm(keras_preactivation_layer, keras_batchnorm_layer)`

   Merges the parameters of a batch normalization layer into the layer
   before it.

   ### Arguments
  keras_preactivation_layer: Layer directly preceding the
  batchnorm layer. 
  
  keras_batchnorm_layer: The batch normalization
  layer to be merged into the preactivation layer.

   ### Returns
  new_layer, (new_weights, new_biases) : Where new_layer is a
  keras layer of the same configuration as
  keras_preactivation_layer, and (new_weights, new_biases) are the
  updated weights and biases for the new_layer to be used with
  `new_layer.set_weights(((new_weights, new_biases)))` after the new
  model is built.
