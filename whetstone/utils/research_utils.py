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

from whetstone.callbacks import Sharpener, WhetstoneLogger
from .utils import load_model

try:
    import cv2
except:
    print('opencv-python is not installed.')
    print('Some functionality will be disabled including: resize_pad(), ...')


def test_sharpening_topology(epochs, datagen, cb_build_model, sharpener, logger, logdir,
                             runs=30, other_callbacks=[], save_model=False, verbose=True):
    """
    Initializes, trains, and tests a sharpenable model a number of times specified by ``runs``
    using ``sharpener`` to sharpen the model and ``logger`` to log data for each run.
    Results of each run are stored in a separate subdirectory automatically created under ``logdir``.
    This function was created to help simplify a typical research workflow.

    Parameters
    ----------
    runs : Number of times to retrain model from scratch and log results.
    epochs: Maximum number of epochs to train ``model`` on each run.
    datagen: A keras Sequence object that should generate batches from the dataset (https://keras.io/utils/).
    cb_build_model : A user-defined callback function that builds and returns a sharpenable sequential keras model.
    sharpener : A callback that is a subclass of Sharpener which will be used to sharpen the model.
    logger : A WhetstoneLogger callback that will handle logging for each run.
    other_callbacks : An optional list of keras callbacks in addition to the sharpener and logger.
    logdir : Directory where logger data will be saved for each run (under separate subfolders).
    save_model : If ``True`` will save the model that resulted from each run.
    verbose : If ``True`` will print basic aggregate statistics at the end of each run.
    """
    assert type(runs) is int and runs > 0
    assert type(epochs) is int and epochs > 0
    assert isinstance(datagen, keras.utils.data_utils.Sequence)
    assert callable(cb_build_model)
    assert isinstance(sharpener, Sharpener)
    assert isinstance(logger, WhetstoneLogger)
    assert other_callbacks == [] or all([isinstance(i, keras.callbacks.Callback) for i in other_callbacks])
    assert os.path.exists(logdir) and os.path.isdir(logdir) # TODO Replace assert with error handling code.
    assert type(save_model) is bool
    assert type(verbose) is bool
    for run_idx in range(runs):
        if verbose:
            print('\nrun:', run_idx)
            ts_start = time.time()
        model = cb_build_model()
        # assert isinstance(model, keras.models.Sequential)
        run_path = os.path.join(logdir, 'run_'+str(run_idx))
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        logger.logdir = run_path
        model.fit_generator(generator = datagen,
                            epochs = epochs,
                            verbose = 1,
                            callbacks = [sharpener, logger] + other_callbacks,
                            use_multiprocessing = True)
        if save_model or run_idx == 0: # always save the model at least once so topology is known.
            model.save(os.path.join(run_path, 'model.h5'))
            try:
                keras.utils.vis_utils.plot_model(model, to_file=os.path.join(run_path, 'model.png'), show_shapes=True, show_layer_names=True)
            except:
                print('plot_model failed. Check dependencies.')
        K.clear_session()
        if verbose:
            print_aggregate_stats(log_dir = logdir)
            run_duration = time.time() - ts_start
            remaining_time = (runs - run_idx - 1) * run_duration
            print('run duration =', round(run_duration, 2), 'seconds.')
            print('estimated time remaining =', round(remaining_time, 2), 'seconds.')


def print_epoch_log(logs):
    """
    Prints the log for a single epoch of whetstone training.
    Can be run stand-alone on the content of the standard log.json file in each epoch folder.

    Parameters
    ----------
    logs : dictionary obtained from reading log.json
    """
    print('')
    print('sharpness                      :', [round(i,3) for i in logs['sharpness']])
    print('training      (loss, accuracy) :', round(logs['train_loss'],3), round(logs['train_accuracy'],3))
    if 'test_loss' in logs and 'test_accuracy' in logs:
        print('testing       (loss, accuracy) :', round(logs['test_loss'],3), round(logs['test_accuracy'],3))
    if 'test_loss_spiking' in logs and 'test_accuracy_spiking' in logs:
        print('testing spike (loss, accuracy) :', round(logs['test_loss_spiking'],3), round(logs['test_accuracy_spiking'],3))


def _read_logs(logdir):
    """
    Reads logs from directory structure created by test_sharpening_topology() and
    returns them as a list of lists, where the outer list is ``runs`` and the inner lists are ``epochs``.

    Parameters
    ----------
    logdir : root directory of logs recorded by test_sharpening_topology().

    Returns
    -------
    A list of lists, where the outer list is ``runs`` and the inner lists are ``epochs``.
    """
    if not (os.path.exists(logdir) and os.path.isdir(logdir)):
        print('Error: print_aggregate_stats:', log_dir, 'does not exist.')
        return []
    runs = []
    for run_idx in range(len(os.listdir(logdir))):
        run_path = os.path.join(logdir, 'run_'+str(run_idx))
        if not (os.path.exists(run_path) and os.path.isdir(run_path)):
            continue
        epochs = []
        for epoch_idx in range(len(os.listdir(run_path))):
            epoch_path = os.path.join(run_path, 'epoch_'+str(epoch_idx))
            if not (os.path.exists(epoch_path) and os.path.isdir(epoch_path)):
                continue
            log_json_path = os.path.join(epoch_path, 'log.json')
            with open(log_json_path, 'rb') as f:
                log = json.load(f)
                epochs.append(log)
        runs.append(epochs)
    return runs


def _read_sharpener_params(log_dir):
    """
    """
    # Below assumes sharpener parameters are the same for all runs.
    sharp_params_path = os.path.join(log_dir, 'run_0/sharpener_params.pkl')
    if not os.path.exists(sharp_params_path):
        # Should probably throw an error here. TODO
        return None
    with open(sharp_params_path, 'rb') as f:
        params = pickle.load(f)
        return params


def print_model(filepath):
    """
    Loads and prints the Keras/Whetstone model from a .h5 file at ``filepath``.

    Parameters
    ----------
    filepath : Path to Keras/Whetstone model which should be a .h5 file produced by model.save(filepath).
    """
    model = load_model(filepath)
    if model is not None:
        model.summary()
    else:
        print('load_model(filepath) returned None.')


def _read_model(log_dir):
    """
    """
    model_path = os.path.join(log_dir, 'run_0/model.h5')
    return load_model(model_path)


def merge_logs(from_dir1, from_dir2, to_dir):
    """
    Use this to merge two directory trees produced by the test_sharpening_topology function.
    Makes it easier to do repeated runs of the same model in parallel on different computers.
    For example, if there were 30 runs under ``from_dir1`` named run_0 -> run_29 and 40 under
    ``from_dir2`` named run_0 -> run_39, ``to_dir`` would end up containing run_0 -> run_69
    where run_30 -> run_69 are renamed copies of run_0 -> run_39 from ``from_dir2``.
    It is up to the user to ensure all runs use the same model and sharpener parameters.
    """
    if not all([os.path.exists(path) and os.path.isdir(path) for path in [from_dir1, from_dir2]]):
        print('Error: First two paths should be directories that exist.')
        return None
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    # --------------------------------
    def _subpaths(path):
        return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    def _copy(src, dst):
        try:
            shutil.copytree(src, dst)
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(src, dst)
            else: raise
    # --------------------------------
    for idx, path in enumerate(_subpaths(from_dir1) + _subpaths(from_dir2)):
        _copy(path, os.path.join(to_dir, 'run_'+str(idx)))


def compress_log(log_dir):
    """
    Use this to aggregate all log.json files in a directory produced by the test_sharpening_topology function.
    Aggregate log is saved as a new .json file in the current working directory.
    Larger files like the model.h5 and activations are omitted to keep the file small.
    If you want everything in the compressed file, then instead use a command line utility like "tar"
    """
    runs = _read_logs(logdir=log_dir)
    fname = os.path.split(os.path.abspath(os.path.normpath(log_dir)))[1]+'.json'
    with open(fname, 'wb') as f:
        json.dump(runs, f, indent=4)


def _calc_line_summaries(runs):
    """
    """
    runs_results = [] # store results for all runs
    for run in runs:
        result = {} # store results for this run
        # Use sharpness deltas to determine which layer was sharpened during each epoch.
        sharpness = [epoch['sharpness'] for epoch in run]
        deltas = [np.array(sharp1)-np.array(sharp0) for sharp0, sharp1 in zip(sharpness, sharpness[1:]+sharpness[-1])]
        line_summary = []
        for delta in deltas:
            if all(delta == 0.0):
                line_summary.append('###')
            else:
                line_summary.append(str(np.argmax(delta)).zfill(3))
        result['line_summary_sharp_delta'] = ' '.join(line_summary)
        # Line summaries for training loss, test accuracy, and spiking test accuracy.
        def summarize(field):
            vals = [epoch[field] for epoch in run]
            return ' '.join([str(int(min(round(v, 3), 0.999)*1000)).zfill(3) for v in vals])
        result['line_summary_train_loss'] = summarize('train_loss')
        result['line_summary_test_acc'] = summarize('test_accuracy')
        result['line_summary_spiking_test_acc'] = summarize('test_accuracy_spiking')
        runs_results.append(result)
    return runs_results


def _calc_aggregate_stats(runs):
    """
    """
    stats = {}
    final_sharpness = [run[-1]['sharpness'] for run in runs]
    finished_sharpening = [all([i == 1.0 for i in sharp]) for sharp in final_sharpness]
    stats['num_runs'] = len(runs)
    stats['num_finished'] = sum(finished_sharpening)
    if stats['num_finished'] == 0:
        return None
    runs_finished = [run for run, finished in zip(runs, finished_sharpening) if finished]
    final_accuracies = [run[-1]['test_accuracy_spiking'] for run in runs_finished]
    final_losses = [run[-1]['train_loss'] for run in runs_finished]
    # ----------------------------
    def descriptive_stats(items, name):
        stats['raw_'+name] = sorted(items, reverse=True)
        stats['mean_'+name] = np.mean(items)
        stats['std_'+name] = np.std(items) # standard deviation
        stats['avg_dev_'+name] = np.mean([abs(i-stats['mean_'+name]) for i in items]) # mean absolute deviation
        stats['max_'+name], stats['min_'+name] = max(items), min(items)
        stats['median_'+name] = np.median(items)
        stats['med_dev_'+name] = np.median([abs(i-stats['median_'+name]) for i in items])
    # ----------------------------
    runs_sharp = [[np.mean(epoch['sharpness']) for epoch in run] for run in runs_finished]
    started = [max([idx for idx, epoch_sharp in enumerate(run) if epoch_sharp == 0.0])+1 for run in runs_sharp]
    ended = [min([idx for idx, epoch_sharp in enumerate(run) if epoch_sharp == 1.0]) for run in runs_sharp]
    pre_sharp_acc = [run[start_idx]['test_accuracy'] for start_idx, run in zip([i-1 for i in started], runs_finished)]
    degradation = [post-pre for pre, post in zip(pre_sharp_acc, final_accuracies)]
    descriptive_stats(final_accuracies, 'spiking_acc') # final spiking accuracy.
    descriptive_stats(final_losses, 'spiking_loss') # final spiking training loss.
    descriptive_stats(started, 'start') # epoch on which sharpening started.
    descriptive_stats(ended, 'end') # epoch on which sharpening ended.
    descriptive_stats(pre_sharp_acc, 'acc_pre') # test accuracy at end of epoch just prior to sharpening start.
    descriptive_stats(degradation, 'deg') # loss in test accuracy relative to pre-sharpening.
    # ---- make line summaries ----
    worst_run = runs_finished[np.array(final_accuracies).argmin()]
    median_run = runs_finished[sorted(enumerate(final_accuracies), key=lambda x:x[1])[len(final_accuracies)/2][0]]
    best_run = runs_finished[np.array(final_accuracies).argmax()]
    (stats['ls_worst'], stats['ls_med'], stats['ls_best']) = _calc_line_summaries([worst_run, median_run, best_run])
    return stats


def _print_aggregate_stats(stats):
    """
    """
    if stats is None:
        print('None of the runs finished.')
        return
    # --------------------------------
    def print_stats(name, percent=False):
        form = (lambda x: str(round(x*100.0,2))+'%') if percent else (lambda x: x if type(x) is int else round(x,4))
        print('  Mean                    =', form(stats['mean_'+name]))
        print('  Med.                    =', form(stats['median_'+name]))
        print('  Min                     =', form(stats['min_'+name]))
        print('  Max                     =', form(stats['max_'+name]))
        print('  Standard Dev.           =', form(stats['std_'+name]))
        print('  Mean Abs Dev.           =', form(stats['avg_dev_'+name]))
        print('  Med. Abs Dev. From Med. =', form(stats['med_dev_'+name]))
        print('  Raw:', [i if type(i) is int else round(i, 4) for i in stats['raw_'+name]])
    # --------------------------------
    print('\nSharpening Statistics:')
    print('\nNumber of runs that did not finish sharpening:', stats['num_runs']-stats['num_finished'])
    print('  N                       =', str(len(stats['raw_spiking_acc'])))
    print('\nFinal Spiking Training Loss Stats:')
    print_stats(name='spiking_loss', percent=False)
    print('\nFinal Spiking Test Accuracy Stats:')
    print_stats(name='spiking_acc', percent=True)
    print('\nPre-Sharpening Test Accuracy Stats:')
    print_stats(name='acc_pre', percent=True)
    print('\nDegradation Stats:')
    print_stats(name='deg', percent=True)
    print('\nSharpening Start Stats:')
    print_stats(name='start', percent=False)
    print('\nSharpening End Stats:')
    print_stats(name='end', percent=False)
    # --------------------------------
    def print_line_summary(run):
        for i in range(int(math.ceil(len(run['line_summary_sharp_delta'])/80.0))):
            print('Layer         :', run['line_summary_sharp_delta'][i*80:(i+1)*80])
            print('Train Loss    :', run['line_summary_train_loss'][i*80:(i+1)*80])
            print('Test Acc      :', run['line_summary_test_acc'][i*80:(i+1)*80])
            print('Spike Test Acc:', run['line_summary_spiking_test_acc'][i*80:(i+1)*80])
            print('')
    # --------------------------------
    print('\nLine Summary Best:')
    print_line_summary(stats['ls_best'])
    print('\nLine Summary Median:')
    print_line_summary(stats['ls_med'])
    print('\nLine Summary Worst:')
    print_line_summary(stats['ls_worst'])


def print_aggregate_stats(log_dir, accuracy_graph=False, loss_graph=False):
    """
    Prints descriptive statistics aggregated over multiple runs recorded by test_sharpening_topology().
    Can also print various optional charts.

    Parameters
    ----------
    log_dir : root directory of logs recorded by test_sharpening_topology().
    accuracy_graph : if ``True`` print line graph showing central tendency and dispersion of accuracy over runs by epoch.
    loss_graph : if ``True`` print line graph showing central tendency and dispersion of loss over runs by epoch.
    """
    runs = _read_logs(logdir = log_dir)
    stats = _calc_aggregate_stats(runs=runs)
    _print_aggregate_stats(stats=stats)


def print_log_summary(log_dir, accuracy_graph=False, loss_graph=False):
    """
    """
    params = _read_sharpener_params(log_dir=log_dir)
    runs = _read_logs(logdir = log_dir)
    if runs == [] or params is None:
        # Should probably throw an error here. TODO
        return None
    model = _read_model(log_dir = log_dir)
    if model is not None:
        model.summary()
    print('\nSharpener Params:')
    for key, value in sorted(params.items()):
        print('  '+key+':', value)
    stats = _calc_aggregate_stats(runs=runs)
    _print_aggregate_stats(stats=stats)


def resize_pad(img, dim, interp=None):
    """ Resizes an opencv image to fit within a square with width/height of ``dim``.

    Images are padded with zeros to fit, not cropped.

    # Arguments
        img: OpenCV image which is a numpy array of type np.uint8.
            Can be of shape (<height>, <width>) or (<height>, <width>, 3)
        dim: Integer, height and width of new square image to be returned.
    """
    h, w = img.shape[0:2]
    img_zeros = np.zeros((dim, dim, 3) if len(img.shape) == 3 else (dim, dim), np.uint8)
    resize_percent = float(dim) / max(h, w)
    w_, h_ = int(math.ceil(w*resize_percent)), int(math.ceil(h*resize_percent))
    if interp is None:
    	img_resized = cv2.resize(img, (w_, h_))
    else:
        img_resized = cv2.resize(img, (w_, h_), interpolation=interp)
    img_zeros[(dim-h_)/2:(dim-h_)/2+h_, (dim-w_)/2:(dim-w_)/2+w_] = img_resized[:,:]
    return img_zeros


def read_img(path, size=None, color=None, filt=None, thresh=None, normalize=True):
    """
    Loads an image from ``path`` and performs optional resizing, colorspace, filtering,
    and thresholding operations (in that order). Returns the image as a numpy array.

    Parameters
    ----------
    path : location of image to be loaded.
    size : tuple (height, width) specifying the resizing opp applied to each image as it's loaded.
    color : optional colorspace conversion to apply after optional resizing. ('gray' or 'bgr')
    filt : optional filter to apply after previous opps.
        should be a tuple of the form (<opp>, arg0, arg1(optional))
        where ``opp`` can be in ('canny', 'dog', or 'sobel')
        The number and type of args depends on the opp. (see code below and opencv documentation)
    thresh : optional thresholding opp to apply after previous opps. (values range from 0.0 to 1.0)
    normalize : If ``True`` returns image as np.float32 with all values between 0.0 and 1.0.
        Otherwise returns as np.uint8 with values between 0 and 255.

    Returns
    -------
    The image as a numpy array.
    """
    assert size is None or (type(size) is tuple and len(size) == 2 and all([type(i) is int for i in size]))
    assert color is None or (type(color) is str and color in ['gray', 'bgr'])
    assert filt is None or (type(filt) is tuple and filt[0] in ['canny', 'sobel', 'dog'])
    assert thresh is None or (type(thresh) is float and thresh >= 0.0 and thresh <= 1.0)
    try:
        img = cv2.imread(path)
    except:
        print('Error: read_img(path =', path, ')')
        return None
    if size is not None and type(size) is tuple and len(size) == 2:
        img = cv2.resize(img, size)
    if filt in ['canny', 'dog', 'sobel']:
        color = 'gray'
    if color is not None and color == 'gray' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color is not None and color == 'bgr' and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    filter_ = filt[0] if type(filt) is tuple else None
    if filter_ == 'canny':
        arg0, arg1 = 100, 200
        if len(filt) == 3:
            arg0, arg1 = int(filt[1]), int(filt[2])
        img = cv2.Canny(img, arg0, arg1)
    elif filter_ == 'dog':
        blur0, blur1 = 5, 3
        if len(filt) == 3:
            blur0, blur1 = int(filt[1]), int(filt[2])
        blur0 += 1 if blur0 % 2 == 0 else 1
        blur1 += 1 if blur1 % 2 == 0 else 1
        blur0_img = cv2.GaussianBlur(img, (blur0, blur0), 0)
        blur1_img = cv2.GaussianBlur(img, (blur1, blur1), 0)
        dog = blur0_img - blur1_img
        img = dog
    elif filter_ == 'sobel':
        ksize_ = 5
        if len(filt) == 2:
            ksize_ = int(filt[1])
        ksize_ += 1 if ksize_ % 2 == 0 else 1
        sobel = cv2.Sobel(img, cv2.CV_8U, dx=1, dy=1, ksize=ksize_)
        img = sobel
    if normalize:
        (img_rows, img_cols) = (img.shape[0], img.shape[1])
        if len(img.shape) == 3:
            img_chnls = img.shape[2]
        else:
            img_chnls = 1
        # Reshape images depending on backend data format.
        if K.image_data_format() == 'channels_first':
            img = img.reshape((img_chnls, img_rows, img_cols))
        else:
            img = img.reshape((img_rows, img_cols, img_chnls))
    img = img.astype(dtype=np.float32)/255.0
    if thresh is not None:
        thresh = float(thresh)
        img = (img >= thresh).astype(np.float32)
    if not normalize:
        img = (img*255.0).astype(dtype=np.uint8)
    return img


def load_img_dataset(path, size=None, color=None, filt=None, thresh=None):
    """
    Loads an image dataset from a specific directory structure under ``path`` which requires
    there to be a /training and /testing folder with subfolders for each class underneath.
    This is a fairly inefficient way to do things that was mainly included for backwards compatibility.

    Parameters
    ----------
    path : path to folder that contains /training and /testing.
    size : tuple (height, width) specifying the resizing opp applied to each image as it's loaded.
    color : optional colorspace conversion to apply after optional resizing. ('gray' or 'bgr')
    filt : optional filter to apply after previous opps. ('canny', 'dog', or 'sobel')
    thresh : optional thresholding opp to apply after previous opps. (values range from 0.0 to 1.0)

    Returns
    -------
    A tuple (num_classes, input_shape, dataset), where num_classes is an int specifying the
    number of classes, input_shape is the shape of the numpy arrays of the loaded images, and
    dataset is a dictionary of the form {'x_train':[...], 'x_test':[...], 'y_train':[...], 'y_test':[...]}
    """
    # Inner function to load classes for testing or training folder.
    # Get names of subdirectories in path. Not recursive.
    def _subdirectories(path):
        return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    # Get names of image files in path. Not recursive.
    def _img_files(path):
        formats = ['jpg','JPG','jpeg','JPEG','png','gif','pgm','tiff','bmp']
        return [n for n in os.listdir(path) if (not os.path.isdir(os.path.join(path, n)) and n.split('.')[-1] in formats)]
    def load_classes(class_names, path):
        x, y = [], []
        for idx, class_ in enumerate(sorted(class_names)):
            class_path = os.path.join(path, class_)
            imgs = _img_files(class_path)
            for img_name in imgs:
                img_path = os.path.join(class_path, img_name)
                img = read_img(img_path, size, color, filt, thresh)
                if img is not None:
                    x.append(img)
                    y.append(idx)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape((len(y), 1))
        return x, y
    x_train, y_train, x_test, y_test = [], [], [], []
    num_classes, input_shape = 0, None
    dataset = {'x_train':None, 'x_test':None, 'y_train':None, 'y_test':None}
    path_train = os.path.join(path, 'training')
    path_test = os.path.join(path, 'testing')
    if os.path.exists(path_train) and os.path.exists(path_test):
        train_classes = _subdirectories(path_train)
        test_classes = _subdirectories(path_test)
        if train_classes == [] or train_classes != test_classes:
            print('Error: load_dataset(): empty or training_classes != testing_classes.')
            return (num_classes, None, None)
        num_classes = len(train_classes)
        x_train, y_train = load_classes(train_classes, path_train)
        x_test, y_test = load_classes(test_classes, path_test)
        dataset['x_train'], dataset['x_test'] = x_train, x_test
        input_shape = dataset['x_train'][0].shape
        dataset['y_train'] = keras.utils.to_categorical(y_train, num_classes)
        dataset['y_test'] = keras.utils.to_categorical(y_test, num_classes)
        return (num_classes, input_shape, dataset)
    else:
        print('Error: load_dataset(): expected training and testing folders.')
        return (num_classes, None, None)
