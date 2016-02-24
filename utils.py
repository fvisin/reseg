from collections import OrderedDict
import os
import sys

import numpy as np
from retrying import retry
from skimage import img_as_float
from sklearn.metrics import confusion_matrix
from skimage.color import label2rgb
from skimage.io import imsave
import theano

from config_datasets import colormap_datasets

floatX = theano.config.floatX


def iterate_minibatches(inputs, targets, batchsize, rng=None, shuffle=False):
    '''Batch iterator
    This is just a simple helper function iterating over training data in
    mini-batches of a particular size, optionally in random order. It assumes
    data is available as numpy arrays. For big datasets, you could load numpy
    arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    own custom data iteration function. For small datasets, you can also copy
    them to GPU at once for slightly improved performance. This would involve
    several changes in the main program, though, and is not demonstrated here.
    '''
    assert len(inputs) == len(targets)
    if shuffle:
        if rng is None:
            raise Exception("A Numpy RandomState instance is needed!")
        indices = np.arange(len(inputs))
        rng.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], excerpt


def save_image(outpath, img):
    import errno
    try:
        os.makedirs(os.path.dirname(outpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass
    imsave(outpath, img)


def validate(f_pred,
             data,
             batchsize,
             nclasses=2,
             samples_ids=-1,
             dataset='camvid',
             saveto='test_lasagne',
             mean=None, std=None, fullmasks=None,
             void_is_present=False,
             filenames=None, folder_dataset='pred'):
    """validate

    Parameters
    ----------
    f_pred :
        The theano function that make the prediction
    data :
        The
    batchsize :
        The
    nclasses :
        The
    samples_ids :
        The
    dataset :
        The
    saveto :
        The
    mean :
        The
    std :
        The
    fullmasks :
        The
    void_is_present :
        In some dataset there are some unlabeled pixels that we don't consider
        in the evalution metrics
    filenames :
        The
    folder_dataset :
        Where to save the segmentation masks. Has to be either 'train',
        'valid' or 'test'

    Returns
    -------
    The function returns the following performance indexes computed on the
    inout dataset:
        * Global Pixel Accuracy
        * Confusion Matrix
        * Mean Class Accuracy (Mean of the diagonal of Norm Conf Matrix)
        * Intersection Over Union Indexes for each class
        * Intersection Over Union Index
    """
    print >>sys.stderr, 'Prediction {}: '.format(folder_dataset),

    if samples_ids.size > 0:
        name = dataset
        seg_path = os.path.join('segmentations', name,
                                saveto.split('/')[-1][:-4])
        # gt_path = os.path.join('gt', name, saveto.split('/')[-1][:-4])
        # img_path = os.path.join('img', name, saveto.split('/')[-1][:-4])
        colormap = colormap_datasets[name]

    inputs, targets = data
    conf_matrix = np.zeros([nclasses, nclasses]).astype('float32')

    im_idx = 0
    for minibatch in iterate_minibatches(inputs,
                                         targets,
                                         batchsize,
                                         shuffle=False):
        mini_x, mini_y, mini_idx = minibatch
        mini_x = img_as_float(mini_x)
        mini_f = filenames[mini_idx]
        preds = f_pred(mini_x.astype(floatX))

        # Compute the confusion matrix for each image
        cf_m = confusion_matrix(mini_y.flatten(), preds.flatten(),
                                range(0, nclasses))
        conf_matrix += cf_m

        # Save samples
        if samples_ids.size > 0:
            for pred, x, y, f in zip(preds, mini_x, mini_y, mini_f):
                if (im_idx in samples_ids or (len(samples_ids) == 1 and
                                              samples_ids == [-1])):
                    # TODO fix daimler dataset --> Marco fix the dataset!
                    # f = f.replace(".pgm", ".png")
                    # save Image + GT + prediction
                    im_name = os.path.basename(f)
                    pred_rgb = label2rgb(pred, colors=colormap)
                    y_rgb = label2rgb(y, colors=colormap)
                    concat_img = np.concatenate((x, y_rgb, pred_rgb), axis=1)
                    outpath = os.path.join(seg_path, folder_dataset, im_name)
                    save_image(outpath, concat_img)
                im_idx += 1

    # Compute metrics
    if void_is_present:
        conf_matrix = conf_matrix[:-1, :-1]

    # Compute per class metrics
    per_class_TP = np.diagonal(conf_matrix)
    per_class_FP = conf_matrix.sum(axis=0) - per_class_TP
    per_class_FN = conf_matrix.sum(axis=1) - per_class_TP

    n_pixels = np.sum(conf_matrix)
    global_acc = per_class_TP.sum() / float(n_pixels)

    # Class Accuracy
    class_acc = per_class_TP / (per_class_FN + per_class_TP)
    class_acc = np.nan_to_num(class_acc)
    mean_class_acc = np.mean(class_acc)

    # Class Intersection over Union
    iou_index = per_class_TP / (per_class_TP + per_class_FP + per_class_FN)
    iou_index = np.nan_to_num(iou_index)
    mean_iou_index = np.mean(iou_index)

    print >>sys.stderr, 'Done'

    return (global_acc, conf_matrix, mean_class_acc, iou_index, mean_iou_index)


def zipp(vparams, params):
    """Copy values from one dictionary to another.

    It will copy all the values from the first dictionary to the second
    dictionary.

    Parameters
    ----------
    vparams : dict
        The dictionary to read the parameters from
    params :
        The dictionary to write the parameters to
    """
    for kk, vv in vparams.iteritems():
        params[kk].set_value(vv)


def unzip(zipped, prefix=None):
    """Return a dict of values out of a dict of theano variables

    If a prefix is provided it will attach the prefix to the name of the
    keys in the dictionary

    Parameters
    ----------
    zipped : dict
        The dictionary of theano variables
    prefix : string, optional
        A prefix to be added to the keys of dictionary
    """
    prefix = '' if prefix is None else prefix + '_'
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[prefix + kk] = vv.get_value()
    return new_params


def unroll(deep_list):
    """ Unroll a deep list into a shallow list

    Parameters
    ----------
    deep_list : list or tuple
        An annidated list of lists and/or tuples. Must not be empty.

    Note
    ----
    The list comprehension is equivalent to:
    ```
    if type(deep_list) in [list, tuple] and len(deep_list):
        if len(deep_list) == 1:
            return unroll(deep_list[0])
        else:
            return unroll(deep_list[0]) + unroll(deep_list[1:])
    else:
        return [deep_list]
    ```
    """
    return ((unroll(deep_list[0]) if len(deep_list) == 1 else
            unroll(deep_list[0]) + unroll(deep_list[1:]))
            if type(deep_list) in [list, tuple] and len(deep_list) else
            [deep_list])


def retry_if_io_error(exception):
    """Return True if IOError.

    Return True if we should retry (in this case when it's an IOError),
    False otherwise.
    """
    print "Filesystem error, retrying in 2 seconds..."
    return isinstance(exception, IOError)


@retry(stop_max_attempt_number=10, wait_fixed=2000,
       retry_on_exception=retry_if_io_error)
def save_with_retry(saveto, args):
    np.savez(saveto, *args)


def ceildiv(a, b):
    """Division rounded up

    Parameters
    ----------
    a : number
        The numerator
    b : number
        The denominator

    Reference
    ---------
    http://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent\
-of-operator-in-python
    """
    return -(-a // b)


def to_float(l):
    """Converts an iterable in a list of floats

    Parameters
    ----------
    l : iterable
        The iterable to be converted to float
    """
    return [float(el) for el in l]


def to_int(l):
    """Converts an iterable in a list of ints

    Parameters
    ----------
    l : iterable
        The iterable to be converted to float
    """
    return [int(el) for el in l]
