import os
import sys
from PIL import Image
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from skimage.color import label2rgb
from skimage import io
from skimage.io import ImageCollection, imsave
from config_datasets import color_list_datasets
import numpy as np
import theano

floatX = theano.config.floatX

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
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
              shuffle='False',
              dataset='camvid',
              saveto='test_lasagne',
              mean=None, std=None, fullmasks=None,
              void_is_present=True, save_seg=True,
              filenames=None, folder_dataset='pred'):
    """
    :param f_pred: theano function that make the prediction
    :param options: the hyper-parameters of the model
    :param data:
    :param mean:
    :param std:
    :param iterator:
    :param fullmasks:
    :param nclasses:
    :param void_is_present: in some dataset there are some unlabeled pixels
                            that we don't consider in the evalution metrics
    :param save_seg: bool, save or not the predicted segmentation masks
    :param filenames:
    :param folder_dataset: where to save the segmentation mask
                           e.g choose one btw 'train', 'test', 'valid'
    :return: the function returns the following performance indexes computed
             on the inout dataset:
               Global Pixel Accuracy
               Confusion Matrix
               Normalized Confusion Matrix
               Mean Class Accuracy (Mean of the diagonal of Norm Conf Matrix)
               Intersection Over Union Indexes for each class
               Intersection Over Union Index
    """

    print >>sys.stderr, 'Prediction: ',

    if save_seg:
        name = dataset
        seg_path = os.path.join('segmentations', name, saveto.split('/')[-1][:-4])
        # gt_path = os.path.join('gt', name, saveto.split('/')[-1][:-4])
        # img_path = os.path.join('img', name, saveto.split('/')[-1][:-4])
        color_list = color_list_datasets[name]

    inputs, targets = data
    conf_matrix = np.zeros([nclasses, nclasses])
    for minibatch in iterate_minibatches(inputs,
                                         targets,
                                         batchsize,
                                         shuffle=False):
        x, y, mini_idx = minibatch
        f = filenames[mini_idx]
        preds = f_pred(x.astype(floatX))

        print >>sys.stderr, '.',

        # computing the confusion matrix for each image
        cf_m = confusion_matrix(y.flatten(), preds.flatten(),
                                range(0, nclasses))
        conf_matrix += cf_m

        if save_seg:
            # save each image of the validation minibatch...
            for im_pred, mini_x, mini_y, filename in zip(preds, x, y, f):
                # fix for daimler dataset
                filename = filename.replace(".pgm", ".png")
                # save segmentation
                base = os.path.basename(filename)

                outpath = os.path.join(seg_path, folder_dataset, base)
                save_image(outpath, label2rgb(im_pred, colors=color_list))

                # double check: save also gt and img to see if it's correct
                # outpath = os.path.join(gt_path, folder_dataset, base)
                # save_image(outpath, label2rgb(y, colors=color_list))
                #
                # outpath = os.path.join(img_path, folder_dataset, base)
                # save_image(outpath, mini_x)

    # [WARNING] : we don't consider the unlabelled pixels so the last
    #             row or column of the confusion matrix are usually discarded

    # Global Accuracy
    if void_is_present:
        correctly_classified_pxls = np.trace(conf_matrix[0:-1, 0:-1])
        pxls = np.sum(conf_matrix[0:-1, :])
    else:
        correctly_classified_pxls = np.trace(conf_matrix)
        pxls = np.sum(conf_matrix)

    global_acc = correctly_classified_pxls / float(pxls)

    # Class Accuracy
    total_per_class = conf_matrix.sum(axis=1)
    cm_normalized = (conf_matrix.astype('float') /
                     total_per_class[:, np.newaxis])
    cm_normalized = np.nan_to_num(cm_normalized)
    if void_is_present:
        class_acc = cm_normalized.diagonal()[0:-1]
    else:
        class_acc = cm_normalized.diagonal()

    # Mean Class Accuracy
    mean_class_acc = np.mean(class_acc)

    # Intersection over Union index
    FP = (conf_matrix.sum(axis=0) -
          np.diagonal(conf_matrix))
    iou_index = conf_matrix.diagonal().astype('float') / (total_per_class + FP)
    iou_index = np.nan_to_num(iou_index)

    # Mean Intersection over Union
    if void_is_present:
        mean_iou_index = np.mean(iou_index[0:-1])
    else:
        mean_iou_index = np.mean(iou_index)

    print >>sys.stderr, 'Done'

    return (global_acc, conf_matrix,
            cm_normalized, mean_class_acc,
            iou_index, mean_iou_index)


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
