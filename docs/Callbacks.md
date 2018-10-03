# Callbacks


### `class whetstone.callbacks.AdaptiveSharpener(min_init_epochs=10, rate=0.25, cz_rate=0.126, critical=0.75, first_layer_relative_rate=1.0, patience=1, sig_increase=0.15, sig_decrease=0.15, **kwargs)`

   Sharpens a model automatically, using training loss to control the
   process.

   ### Arguments
  min_init_epochs: Integer, minimum number of epochs to train
  before sharpening begins. rate: Float, amount to sharpen a layer
  per epoch. 
  
  cz_rate: Float, rate of sharpening in Critical Zone,
  which is when layer sharpness >= `critical`. 
  
  critical: Float, critical sharpness after which to apply cz_rate.
  
  first_layer_relative_rate: Float, percentage of normal
  sharpening rate to use in first layer. patience: Integer, how
  many epochs to wait for significant improvement. 
  
  sig_increase: Float, percent increase in loss considered significant.
  
  sig_decrease: Float, percent decrease in loss considered
  significant.

`class whetstone.callbacks.RLSharpener(start_step, layer_duration, **kwargs)`

   Experimental Sharpener for use with KerasRL.

   Behaves like the `SimpleSharpener`, but based on steps instead of
   batches or epochs.

   ### Arguments
  start_step: Integer, step to begin sharpening. layer_duration:
  Integer, number of steps over which to sharpen each layer.

### `class whetstone.callbacks.ScheduledSharpener(schedule=None, num_layers=None, start=None, duration=None, intermission=None, **kwargs)`

   Sharpens each layer according to a manually defined schedule.

   Takes a sharpening schedule as input and gradually sharpens on each
   batch by the appropriate amount, as automatically calculated, such
   that each layer begins and ends sharpening as specified in the
   schedule. Note: The first epoch is not allowed to perform any
   sharpening. This is because we need to know the number of batches
   per epoch.

   If schedule isn't passed, then num_layers, start, duration, and
   intermission must be supplied. These will be used to generate a
   schedule (see gen_schedule method).

   ### Arguments
  schedule: List of tuples of the form [(start_epoch, stop_epoch),
  (start_epoch, stop_epoch), …]
 specifying for which epoch to to begin and end sharpening for
 each spiking layer, where the sharpening schedule for the ith
 spiking layer would be the ith tuple in the list. Note that
 the first epoch is 0, not 1.

  num_layers: Integer, number of sharpenable layers in the model.
  start: Integer, epoch number on which to begin sharpening.
  duration: Integer, number of epochs over which to sharpen each
  layer. intermission: Integer, number of epochs to halt
  sharpening between layers.

   #### gen_schedule(num_layers, start, duration, intermission)

  Generates a sharpening schedule for use with `ScheduledSharpener`.

  ##### Arguments
 num_layers: Integer, number of sharpenable layers in the
 model. 
 
 start: Integer, epoch number on which to begin
 sharpening. 
 
 duration: Integer, number of epochs over which to
 sharpen each layer. intermission: Integer, number of epochs
 to halt sharpening between layers.

  ### Returns
 List of tuples of the form [(start_epoch, stop_epoch),
 (start_epoch, stop_epoch), …] specifying for which epoch to
 to begin and end sharpening for each spiking layer, where the
 sharpening schedule for the ith spiking layer would be the
 ith tuple in the list.

### `class whetstone.callbacks.Sharpener(bottom_up=True, verbose=False)`

   Absract base class used for different sharpening callbacks.

   ### Arguments
  bottom_up: Boolean, if `True`, sharpens one layer at a time,
 sequentially, starting with the first. If `False`, sharpens
 all layers uniformly.

  verbose: Boolean, if `True`, prints status updates during
  training.

   #### set_layer_sharpness(values)

  Sets the sharpness values of all spiking layers.

  ##### Arguments
 values: A list of sharpness values (between 0.0 and 1.0
 inclusive) for each
spiking layer in the same order as their indices.

   #### set_model_sharpness(value)

  Sets the sharpness of the whole model either in a bottom_up or
  uniform fashion depending on the
 value of the bottom_up instance variable.

  ##### Arguments
 value: Float, value between 0.0 and 1.0 inclusive that
 specifies the sharpness of the model.

### `class whetstone.callbacks.SimpleSharpener(start_epoch, steps=4, epochs=True, **kwargs)`

   Basic sharpener that sharpens each layer in a set number of
   batches.

   ### Arguments
  start_epoch: Integer, epoch on which to begin sharpening. steps:
  Integer, number of steps by which each layer should be fully
  sharpened. epochs: Boolean, if `True`, step on each epoch.
  Otherwise, step on each batch.

### `class whetstone.callbacks.WhetstoneLogger(logdir, sharpener=None, test_set=None, log_weights=False)`

   Keras callback that handles logging (not a type of beer).

  Automatically creates a separate subfolder for each epoch.

   ### Arguments
  logdir: Directory in which to log results. sharpener: Reference
  to callback of type `Sharpener`.

 If passed, metadata from the sharpener will be recorded.

  test_set: Test set tuple in form (x_test, y_test).
 If passed, test set accuracy will be evaluated on current and
 fully-sharpened versions of the net at the end of each epoch.

  log_weights: Boolean, if `True`, logs weights of the entire net
  at the end of
 each epoch.
