from __future__ import absolute_import
from __future__ import print_function

import pkg_resources
import keras
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from .layers import Spiking
from .utils.layer_utils import get_spiking_layer_indices, set_layer_sharpness, set_model_sharpness
import os, sys, json, pickle, copy, time


class Sharpener(Callback):
    """Absract base class used for different sharpening callbacks.

    # Arguments
        bottom_up: Boolean, if ``True``, sharpens one layer at a time, 
            sequentially, starting with the first. If ``False``, sharpens all layers uniformly.
        verbose: Boolean, if ``True``, prints status updates during training.
    """
    def __init__(self, bottom_up=True, verbose=False):
        super(Callback, self).__init__()
        assert type(bottom_up) is bool
        assert type(verbose) is bool
        self.bottom_up = bottom_up
        self.verbose = verbose
        self.current_epoch = 0

    def get_config(self):
        config = {'bottom_up':self.bottom_up, 'verbose':self.verbose}
        return config

    def on_train_begin(self, logs=None):
        self.sharpness = [0.0 for _ in range(self._num_spiking_layers())]
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch
        if all([i == 1.0 for i in self.sharpness]):
            self.model.stop_training = True

    def _spiking_layer_indices(self):
        """Returns indices of layers that can be sharpened. """
        return get_spiking_layer_indices(model=self.model)

    def _num_spiking_layers(self):
        """Returns number of layers in self.model that can be sharpened. """
        return len(self._spiking_layer_indices())

    def set_layer_sharpness(self, values):
        """Sets the sharpness values of all spiking layers.

        # Arguments
            values: A list of sharpness values (between 0.0 and 1.0 inclusive) for each 
                spiking layer in the same order as their indices.
        """
        set_layer_sharpness(model=self.model, values=values)
        self.sharpness = values

    def set_model_sharpness(self, value):
        """Sets the sharpness of the whole model either in a bottom_up or uniform fashion depending on the
           value of the bottom_up instance variable.

        # Arguments
            value: Float, value between 0.0 and 1.0 inclusive that specifies the sharpness of the model.
        """
        values = set_model_sharpness(model=self.model, value=value, bottom_up=self.bottom_up)
        self.sharpness = values


class SimpleSharpener(Sharpener):
    """Basic sharpener that sharpens each layer in a set number of batches.

    # Arguments
        start_epoch: Integer, epoch on which to begin sharpening.
        steps: Integer, number of steps by which each layer should be fully sharpened.
        epochs: Boolean, if ``True``, step on each epoch.  Otherwise, step on each batch.
    """
    def __init__(self, start_epoch, steps=4, epochs=True, **kwargs):
        super(SimpleSharpener, self).__init__(**kwargs)
        assert type(start_epoch) is int and start_epoch >= 0
        assert type(steps) is int and steps >= 1
        assert type(epochs) is bool
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.steps = steps 

    def get_config(self):
        config = {'epochs':self.epochs, 'start_epoch':self.start_epoch, 'steps':self.steps}
        base_config = super(SimpleSharpener, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def on_train_begin(self, logs=None):
        super(SimpleSharpener, self).on_train_begin(logs)
        self.total_steps = (self.steps * self._num_spiking_layers()) if self.bottom_up else self.steps
        self.amt = 1.0/float(self.total_steps)
        self.taken_steps = 0
        self.model_sharpness = 0.0

    def _perform_sharpening(self):
        if self.current_epoch >= self.start_epoch:
            self.model_sharpness += self.amt
            self.taken_steps += 1
            if self.taken_steps < self.total_steps:
                self.set_model_sharpness(self.model_sharpness)
            else:
                self.set_model_sharpness(1.0)

    def on_epoch_end(self, epoch, logs=None):
        super(SimpleSharpener, self).on_epoch_end(epoch, logs)
        if (self.epochs):
            self._perform_sharpening()

    def on_batch_end(self, batch, logs=None):
        if (not self.epochs):
            self._perform_sharpening(logs)


class ScheduledSharpener(Sharpener):
    """Sharpens each layer according to a manually defined schedule.

    Takes a sharpening schedule as input and gradually sharpens on each batch by 
    the appropriate amount, as automatically calculated, such that each layer begins
    and ends sharpening as specified in the schedule. Note: The first epoch is not allowed
    to perform any sharpening. This is because we need to know the number of batches per epoch.

    If schedule isn't passed, then num_layers, start, duration, and intermission must be supplied.
    These will be used to generate a schedule (see gen_schedule method).

    # Arguments
        schedule: List of tuples of the form [(start_epoch, stop_epoch), (start_epoch, stop_epoch), ...] 
            specifying for which epoch to to begin and end sharpening for each spiking layer, where the 
            sharpening schedule for the ith spiking layer would be the ith tuple in the list.
            Note that the first epoch is 0, not 1.
        num_layers: Integer, number of sharpenable layers in the model.
        start: Integer, epoch number on which to begin sharpening.
        duration: Integer, number of epochs over which to sharpen each layer.
        intermission: Integer, number of epochs to halt sharpening between layers.
    """
    def __init__(self, schedule=None, num_layers=None, start=None, duration=None, intermission=None, **kwargs):
        super(ScheduledSharpener, self).__init__(**kwargs)
        if schedule is None:
            schedule = self.gen_schedule(num_layers, start, duration, intermission)
        else:
            assert type(schedule) is list
            assert all([type(st) is int and type(sp) is int for (st, sp) in schedule])
            assert all([(sp - st) > 0 for (st, sp) in schedule])
            assert all([sp > 0 and st > 0 for (st, sp) in schedule])
        self.schedule = schedule
        self.batch = 0
        try:
            self.num_batches = num_batches
        except:
            pass

    def gen_schedule(self, num_layers, start, duration, intermission):
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
        
    def get_config(self):
        config = {'schedule':self.schedule}
        try:
            config['batches_per_epoch'] = self.num_batches
        except:
            pass
        base_config = super(ScheduledSharpener, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def on_train_begin(self, logs=None):
        super(ScheduledSharpener, self).on_train_begin(logs)

    def _init_schedule(self):
        self.num_batches = self.batch + 1
        self.incs = [1.0/(((stop-start)*self.num_batches)-1) for (start, stop) in self.schedule]

    def _perform_sharpening(self):
        for idx, layer_idx in enumerate(self._spiking_layer_indices()):
            (start, stop) = self.schedule[idx]
            if self.current_epoch >= start and self.current_epoch < stop:
                self.sharpness[idx] += self.incs[idx]
                self.sharpness[idx] = min(1.0, self.sharpness[idx])
            elif self.current_epoch >= stop:
                self.sharpness[idx] = 1.0
        self.set_layer_sharpness(values=self.sharpness)

    def on_batch_end(self, batch, logs=None):
        if self.current_epoch > 0:
            self._perform_sharpening()
        self.batch = batch

    def on_epoch_end(self, epoch, logs=None):
        super(ScheduledSharpener, self).on_epoch_end(epoch, logs)
        if epoch == 0:
            self._init_schedule()


class RLSharpener(Sharpener):
    """ Experimental Sharpener for use with KerasRL.

        Behaves like the SimpleSharpener, but based on steps instead of batches or epochs.

        # Arguments
            start_step: Integer, step to begin sharpening.
            layer_duration: Integer, number of steps over which to sharpen each layer.
    """
    def __init__(self, start_step, layer_duration, **kwargs):
        super(RLSharpener, self).__init__(**kwargs)
        assert type(start_step) is int and start_step > 0
        assert type(layer_duration) is int and layer_duration > 0
        self.start_step = start_step # step on which to begin sharpening.
        self.layer_duration = layer_duration # how many steps to sharpen each layer over.
        self.episode = 0 # current episode
        self.step = 0 # step at end of previous episode
        self.bottom_up = True

    def get_config(self):
        config = {'start_step':self.start_step, 'layer_duration':self.layer_duration}
        base_config = super(RLSharpener, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def on_train_begin(self, logs=None):
        super(RLSharpener, self).on_train_begin()
    
    def on_episode_end(self, episode, logs):
        # {'episode_reward': 108.0, 'nb_steps': 6402, 'nb_episode_steps': 108}
        self.episode = episode
        self.step = logs['nb_steps']
        if self.step >= self.start_step:
            self._perform_sharpening(step=self.step)
        super(RLSharpener, self).on_epoch_end(1, None) # ensures training halts when all layers sharpened.

    def _perform_sharpening(self, step):
        model_sharpness = min(1.0, max(0.0, int(step) - self.start_step) / float(self.layer_duration) / float(self._num_spiking_layers()))
        print('model_sharpness =', model_sharpness)
        #print('self.model.stop_traininng =', self.model.stop_training)
        self.set_model_sharpness(model_sharpness)


class AdaptiveSharpener(Sharpener):
    """Sharpens a model automatically, using training loss to control the process.

    # Arguments
        min_init_epochs: Integer, minimum number of epochs to train before sharpening begins.
        rate: Float, amount to sharpen a layer per epoch.
        cz_rate: Float, rate of sharpening in Critical Zone, which is when layer sharpness >= ``critical``.
        critical: Float, critical sharpness after which to apply cz_rate.
        first_layer_relative_rate: Float, percentage of normal sharpening rate to use in first layer.
        patience: Integer, how many epochs to wait for significant improvement.
        sig_increase: Float, percent increase in loss considered significant.
        sig_decrease: Float, percent decrease in loss considered significant.
    """
    def __init__(self, min_init_epochs=10, 
                 rate=0.25, 
                 cz_rate=0.126, 
                 critical=0.75, 
                 first_layer_relative_rate=1.0, 
                 patience=1, 
                 sig_increase=0.15, 
                 sig_decrease=0.15, 
                 **kwargs):
        super(AdaptiveSharpener, self).__init__(**kwargs)
        assert type(min_init_epochs) is int and min_init_epochs >= 1
        assert type(rate) is float and rate > 0.0 and rate <= 1.0
        assert type(cz_rate) is float and cz_rate > 0.0 and cz_rate <= 1.0
        assert type(critical) is float and critical >= 0.0 and critical <= 1.0
        assert type(first_layer_relative_rate) is float and first_layer_relative_rate > 0.0
        assert type(patience) is int and patience >= 0
        assert type(sig_increase) is float and sig_increase > 0.0
        assert type(sig_decrease) is float and sig_decrease > 0.0
        self.min_init_epochs = min_init_epochs
        self.rate = rate
        self.cz_rate = cz_rate
        self.critical = critical
        self.first_layer_relative_rate = first_layer_relative_rate
        self.patience = patience
        self.sig_increase = sig_increase
        self.sig_decrease = sig_decrease
        try:
            self.batches_per_epoch = batches_per_epoch
        except:
            pass

    def get_config(self):
        config = {'min_init_epochs':self.min_init_epochs, 
                  'rate':self.rate, 
                  'cz_rate':self.cz_rate,
                  'critical':self.critical,
                  'first_layer_relative_rate':self.first_layer_relative_rate,
                  'patience':self.patience,
                  'sig_increase':self.sig_increase,
                  'sig_decrease':self.sig_decrease,
                 }
        try:
            config['batches_per_epoch'] = self.batches_per_epoch
        except:
            pass 
        base_config = super(AdaptiveSharpener, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def on_train_begin(self, logs=None):
        super(AdaptiveSharpener, self).on_train_begin(logs)
        self.sharpening = False # state variable.
        self.reference_loss = 1000000.0 # loss after last significant change.
        self.epochs_no_improvement = 0  # number of epochs since the loss improved significantly.
        self.batch = 0
        self.batches_per_epoch = None
        self.wait = False
        
    def _perform_sharpening(self, logs=None):
        unfinished_layers = [idx for idx, s in enumerate(self.sharpness) if s < 1.0]
        if len(unfinished_layers) > 0:
            if self.bottom_up:
                if not self.wait:
                    sharpen_idx = min(unfinished_layers)
                    sharpen_amount = self.rate
                    if self.sharpness[sharpen_idx] >= self.critical:
                        sharpen_amount = self.cz_rate
                    if sharpen_idx == 0: # if first spiking layer
                        sharpen_amount *= self.first_layer_relative_rate
                    sharpen_amount *= (1.0/float(self.batches_per_epoch))
                    self.sharpness[sharpen_idx] = min(1.0, self.sharpness[sharpen_idx] + sharpen_amount)
                    if 1.0 - self.sharpness[sharpen_idx] < 0.001:
                        self.sharpness[sharpen_idx] = 1.0
                    if self.sharpness[sharpen_idx] == 1.0:
                        self.wait = True
            else: # uniform sharpen
                sharpen_amount = self.rate
                if self.sharpness[0] >= self.critical:
                    sharpen_amount = self.cz_rate
                sharpen_amount *= (1.0/float(self.batches_per_epoch))
                new_uniform_sharpness = min(1.0, self.sharpness[0] + sharpen_amount)
                if 1.0 - new_uniform_sharpness < 0.000001:
                    new_uniform_sharpness = 1.0
                self.sharpness = [new_uniform_sharpness for _ in range(len(self.sharpness))]
            self.set_layer_sharpness(values=self.sharpness)
        else:
            self.sharpening = False

    def on_epoch_end(self, epoch, logs=None):
        super(AdaptiveSharpener, self).on_epoch_end(epoch, logs)
        self.wait = False # reset overshoot protection flag
        improved, degraded = False, False
        percent_change = (logs['loss'] - self.reference_loss) / self.reference_loss
        if percent_change >= self.sig_increase:
            degraded = True
        elif percent_change <= -self.sig_decrease:
            improved = True
        if self.current_epoch >= self.min_init_epochs - 1:
            if improved:
                self.reference_loss = logs['loss']
                self.epochs_no_improvement = 0
            else: # degraded or remained unchanged
                self.epochs_no_improvement += 1
            if self.sharpening:
                if degraded:
                    self.reference_loss = logs['loss']
                    self.epochs_no_improvement = 0
                    self.sharpening = False
            else: # not sharpening
                if self.epochs_no_improvement > self.patience:
                    self.reference_loss = logs['loss']
                    self.epochs_no_improvement = 0
                    self.sharpening = True
        else: # not time to consider sharpening yet.
            self.reference_loss = logs['loss']
        if epoch == 0:
            self.batches_per_epoch = self.batch + 1
        if self.verbose:
            print('\nloss =', logs['loss'])
            print('current_reference_loss =', self.reference_loss)
            print('percent_change =', percent_change)
            print('improved =', improved, 'degraded =', degraded) 
            print('epochs_not_improved =', self.epochs_no_improvement) 
            print('sharpening =', self.sharpening)
            print('sharpness =', [round(i, 4) for i in self.sharpness])

    def on_batch_end(self, batch, logs=None):
        if self.sharpening:
            self._perform_sharpening(logs)
        self.batch = batch


class WhetstoneLogger(Callback):
    """Keras callback that handles logging (not a type of beer).
       
       Automatically creates a separate subfolder for each epoch.

    # Arguments
        logdir: Directory in which to log results.
        sharpener: Reference to callback of type ``Sharpener``. 
            If passed, metadata from the sharpener will be recorded.
        test_set: Test set tuple in form (x_test, y_test).
            If passed, test set accuracy will be evaluated on current and 
            fully-sharpened versions of the net at the end of each epoch.
        log_weights: Boolean, if ``True``, logs weights of the entire net at the end of 
            each epoch.
    """
    def __init__(self, logdir, 
                 sharpener=None, 
                 test_set=None, 
                 log_weights=False):
        super(Callback, self).__init__()
        assert os.path.exists(logdir) and os.path.isdir(logdir)
        assert sharpener is None or isinstance(sharpener, Sharpener)
        assert test_set is None or (type(test_set) is tuple and len(test_set) == 2)
        assert type(log_weights) is bool
        self.logdir = logdir
        self.sharpener = sharpener
        self.test_set = test_set
        self.log_weights = log_weights

    def on_train_begin(self, logs=None):
        # Create metadata files that store sharpener params and copy of exemplar set.
        with open(os.path.join(self.logdir, 'sharpener_params.pkl'), 'wb') as f:
            pickle.dump(self.sharpener.get_config(), f, protocol=1)
        environ_info = {'time':time.time()}
        try:
            environ_info['whetstone_version'] = pkg_resources.get_distribution('whetstone').version
            environ_info['keras_version'] = keras.__version__
            environ_info['numpy_version'] = np.__version__
            environ_info['python_version'] = sys.version
            environ_info['backend'] = str(K._backend)
            if environ_info['backend'] == 'tensorflow':
                environ_info['tensorflow_version'] = K.tf.__version__
        except:
            pass
        with open(os.path.join(self.logdir, 'environ.pkl'), 'wb') as f:
            pickle.dump(environ_info, f, protocol=1)

    def on_epoch_end(self, epoch, logs=None):
        # Create directory to store logs for the current epoch
        epoch_path = os.path.join(self.logdir, 'epoch_'+str(epoch))
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        # Store general logs in a human-readable form.
        logs_ = {'train_loss':logs['loss'], 'train_accuracy':logs['acc']}
        if self.sharpener is not None:
            logs_['sharpness'] = self.sharpener.sharpness
        if self.test_set is not None:
            (x_test, y_test) = self.test_set
            logs_['test_loss'], logs_['test_accuracy'] = self.model.evaluate(x_test, y_test, verbose=0)[0:2]
            if self.sharpener is not None:
                self.sharpener.set_layer_sharpness(values=[1.0 for _ in logs_['sharpness']])
                logs_['test_loss_spiking'], logs_['test_accuracy_spiking'] = self.model.evaluate(x_test, y_test, verbose=0)[0:2]
                self.sharpener.set_layer_sharpness(values=logs_['sharpness']) # restore
        log_path = os.path.join(epoch_path, 'log.json')
        with open(log_path, 'wb') as f:
            json.dump(logs_, f, indent=4)
        if self.log_weights:
            self.model.save(os.path.join(epoch_path, 'model_epoch_'+str(epoch)+'.h5'))


