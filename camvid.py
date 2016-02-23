from __future__ import division
import os
from collections import OrderedDict
import ctypes
import numpy as np

import scipy.io as sio
from skimage import exposure
from skimage import io
from skimage import img_as_float, img_as_ubyte, img_as_uint, img_as_int
from skimage.color import label2rgb, rgb2hsv, hsv2rgb
from skimage.io import ImageCollection, imsave
from skimage.transform import resize
from itertools import izip

from config_datasets import (colormap_datasets as colors_list)
from helper_dataset import zero_pad, \
    compare_mask_image_filenames, convert_RGB_mask_to_index, \
    rgb2illumination_invariant, save_image

N_DEBUG = -5
DEBUG_SAVE_IMG = False
DEBUG_SAVE_MASK = False

intX = 'uint8'


def properties():
    return {  # 'reshape': [212, 264, 3],
        # 'reorder': [0, 1, 2],
        # 'rereorder': [0, 1, 2]
    }


"""
compare_mask_image_filenames:
    mask = [i.split('/')[-1].replace('_L.png', '.png') for i in filenames_mask]

compare_mask_image_filenames_segnet
    mask = [i.split('/')[-1].replace('annot', '') for i in filenames_mask]
"""


def load_images(img_path, gt_path, colors, load_greylevel_mask=False,
                resize_images=False, resize_size=-1, save=False,
                color_space='RGB'):

    if load_greylevel_mask:
        assert not save
    images = []
    masks = []
    filenames_images = []
    print "Loading images..."
    # print img_path
    labs = ImageCollection(os.path.join(img_path, "*.png"))
    for i, (inpath, im) in enumerate(izip(labs.files, labs)):

        if i == N_DEBUG:
            break

        assert np.amax(im) <= 255, "Image is not 8-bit"
        if resize_images and resize_size != -1:

            w, h = resize_size
            im = resize(im, (h, w), order=3)
            # order=3 : bicubic interpolation
            # it's normalized by default btw 0-1 by the resize function
            # so we want to preserve the range
            im = img_as_ubyte(im)
        im = im.astype(intX)

        if color_space == "HSV":
            im = rgb2hsv(im)

        if DEBUG_SAVE_IMG:
            outpath = inpath.replace('imgs', 'debug_imgs')
            save_image(outpath, im)

        images.append(im)
        filenames_images.append(inpath)

    print "Loading masks..."
    if load_greylevel_mask:
        gt_path = gt_path.replace("gt", "gt_grey")
    filenames_mask = []
    labs = ImageCollection(os.path.join(gt_path, "*.png"))
    for i, (inpath, im) in enumerate(izip(labs.files, labs)):

        if i == N_DEBUG:
            break

        if resize_images and resize_size != -1:
            w, h = resize_size
            im = (resize(im, (h, w), order=0) * 255).astype(np.uint8)
        filenames_mask.append(inpath)
        # print inpath
        if load_greylevel_mask:
            mask = im
        else:
            mask = convert_RGB_mask_to_index(
                im, colors, ignore_missing_labels=True)
            if save:
                outpath = inpath.replace("gt", "gt_grey")
                save_image(outpath, mask)

        mask = np.array(mask).astype(intX)

        if DEBUG_SAVE_MASK:
            outpath = inpath.replace('gt', 'debug_gt')
            outpath = inpath.replace('annot', 'debug_annot')
            print np.unique(mask)

            save_image(outpath, label2rgb(mask, colors=colors_list['camvid']))

        masks.append(mask)

    assert len(images) == len(
        masks), "Train Images and masks are not in the same quantity"
    return images, masks, filenames_images


def load_dataset_camvid(path, load_greylevel_mask=False, classes='subset_11',
                        resize_images=False,
                        resize_size=-1,
                        use_standard_split=True,
                        save=False,
                        color_space='RGB'):
    # WORKING: but image Seq05VD_f02610_L.png has some problems, some pixels
    # have other values so I treated as Void

    img_train_path = os.path.join(path, 'imgs', 'train')
    img_test_path = os.path.join(path, 'imgs', 'test')
    img_val_path = os.path.join(path, 'imgs', 'val')

    gt_train_path = os.path.join(path, 'gt', 'train')
    gt_test_path = os.path.join(path, 'gt', 'test')
    gt_val_path = os.path.join(path, 'gt', 'val')

    camvid_all_colors = OrderedDict([
        ("Animal", np.array([[64, 128, 64]], dtype=np.uint8)),
        ("Archway", np.array([[192, 0, 128]], dtype=np.uint8)),
        ("Bicyclist", np.array([[0, 128, 192]], dtype=np.uint8)),
        ("Bridge", np.array([[0, 128, 64]], dtype=np.uint8)),
        ("Building", np.array([[128, 0, 0]], dtype=np.uint8)),
        ("Car", np.array([[64, 0, 128]], dtype=np.uint8)),
        ("CartLuggagePram", np.array([[64, 0, 192]], dtype=np.uint8)),
        ("Child", np.array([[192, 128, 64]], dtype=np.uint8)),
        ("Column_Pole", np.array([[192, 192, 128]], dtype=np.uint8)),
        ("Fence", np.array([[64, 64, 128]], dtype=np.uint8)),
        ("LaneMkgsDriv", np.array([[128, 0, 192]], dtype=np.uint8)),
        ("LaneMkgsNonDriv", np.array([[192, 0, 64]], dtype=np.uint8)),
        ("Misc_Text", np.array([[128, 128, 64]], dtype=np.uint8)),
        ("MotorcycleScooter", np.array([[192, 0, 192]], dtype=np.uint8)),
        ("OtherMoving", np.array([[128, 64, 64]], dtype=np.uint8)),
        ("ParkingBlock", np.array([[64, 192, 128]], dtype=np.uint8)),
        ("Pedestrian", np.array([[64, 64, 0]], dtype=np.uint8)),
        ("Road", np.array([[128, 64, 128]], dtype=np.uint8)),
        ("RoadShoulder", np.array([[128, 128, 192]], dtype=np.uint8)),
        ("Sidewalk", np.array([[0, 0, 192]], dtype=np.uint8)),
        ("SignSymbol", np.array([[192, 128, 128]], dtype=np.uint8)),
        ("Sky", np.array([[128, 128, 128]], dtype=np.uint8)),
        ("SUVPickupTruck", np.array([[64, 128, 192]], dtype=np.uint8)),
        ("TrafficCone", np.array([[0, 0, 64]], dtype=np.uint8)),
        ("TrafficLight", np.array([[0, 64, 64]], dtype=np.uint8)),
        ("Train", np.array([[192, 64, 128]], dtype=np.uint8)),
        ("Tree", np.array([[128, 128, 0]], dtype=np.uint8)),
        ("Truck_Bus", np.array([[192, 128, 192]], dtype=np.uint8)),
        ("Tunnel", np.array([[64, 0, 64]], dtype=np.uint8)),
        ("VegetationMisc", np.array([[192, 192, 0]], dtype=np.uint8)),
        ("Wall", np.array([[64, 192, 0]], dtype=np.uint8)),
        ("Void", np.array([[0, 0, 0]], dtype=np.uint8))
    ])

    camvid_11_colors = OrderedDict([
        ("Sky", np.array([[128, 128, 128]], dtype=np.uint8)),
        ("Building", np.array([[128, 0, 0],   # Building
                               [64, 192, 0],  # Wall
                               [0, 128, 64]   # Bridge
                               ], dtype=np.uint8)),
        ("Column_Pole", np.array([[192, 192, 128]], dtype=np.uint8)),
        ("Road", np.array([[128, 64, 128],  # Road
                           [128, 0, 192],   # LaneMkgsDriv
                           [192, 0, 64],    # LaneMkgsNonDriv
                           [128, 128, 192]  # RoadShoulder
                           ], dtype=np.uint8)),
        ("Sidewalk", np.array([[0, 0, 192],    # Sidewalk
                               [64, 192, 128]  # ParkingBlock
                               ], dtype=np.uint8)),
        ("Tree", np.array([[128, 128, 0],  # Tree
                           [192, 192, 0]   # VegetationMisc
                           ], dtype=np.uint8)),
        ("SignSymbol", np.array([[192, 128, 128],  # SignSymbol
                                 # [128, 128, 64],   # Misc_Text
                                 [0, 64, 64],      # TrafficLight
                                 [0, 0, 64]        # TrafficCone
                                 ], dtype=np.uint8)),
        ("Fence", np.array([[64, 64, 128]], dtype=np.uint8)),
        ("Car", np.array([[64, 0, 128],     # Car
                          [192, 128, 192],  # Truck_Bus
                          [64, 128, 192],   # SUVPickupTruck
                          [128, 64, 64],    # OtherMoving
                          [64, 0, 192],     # CartLuggagePram
                          ], dtype=np.uint8)),
        ("Pedestrian", np.array([[64, 64, 0],    # Pedestrian
                                 [192, 128, 64]  # Child
                                 ], dtype=np.uint8)),
        ("Bicyclist", np.array([[0, 128, 192],  # Bicyclist
                                [192, 0, 192],  # MotorcycleScooter
                                ], dtype=np.uint8)),
        ("Void", np.array([[0, 0, 0]], dtype=np.uint8))
    ])  # consider as void all the other classes

    camvid_colors = camvid_11_colors if classes == 'subset_11' else \
        camvid_all_colors

    print "Processing Camvid train dataset..."
    img_train, mask_train, filenames_train = load_images(
        img_train_path, gt_train_path, camvid_colors, load_greylevel_mask,
        resize_images, resize_size, save, color_space)

    print "Processing Camvid test dataset..."
    img_test, mask_test, filenames_test = load_images(
        img_test_path, gt_test_path, camvid_colors, load_greylevel_mask,
        resize_images, resize_size, save, color_space)
    print "Processing Camvid validation dataset..."
    img_val, mask_val, filenames_val = load_images(
        img_val_path, gt_val_path, camvid_colors, load_greylevel_mask,
        resize_images, resize_size, save, color_space)

    return (img_train, mask_train, filenames_train,
            img_test, mask_test, filenames_test,
            img_val, mask_val, filenames_val)


def load_dataset_camvid_segnet(path):
    img_train_path = os.path.join(path, 'train')
    img_valid_path = os.path.join(path, 'val')
    img_test_path = os.path.join(path, 'test')

    gt_train_path = os.path.join(path, 'trainannot')
    gt_valid_path = os.path.join(path, 'valannot')
    gt_test_path = os.path.join(path, 'testannot')

    camvid_colors = OrderedDict([
        ("Sky", np.array([128, 128, 128], dtype=np.uint8)),
        ("Building", np.array([128, 0, 0], dtype=np.uint8)),
        ("Column_Pole", np.array([192, 192, 128], dtype=np.uint8)),
        ("Road", np.array([128, 64, 128], dtype=np.uint8)),
        ("Sidewalk", np.array([0, 0, 192], dtype=np.uint8)),
        ("Tree", np.array([128, 128, 0], dtype=np.uint8)),
        ("SignSymbol", np.array([192, 128, 128], dtype=np.uint8)),
        ("Fence", np.array([64, 64, 128], dtype=np.uint8)),
        ("Car", np.array([64, 0, 128], dtype=np.uint8)),
        ("Pedestrian", np.array([64, 64, 0], dtype=np.uint8)),
        ("Bicyclist", np.array([0, 128, 192], dtype=np.uint8)),
        ("Void", np.array([0, 0, 0], dtype=np.uint8))
    ])

    print "Processing Camvid SegNet train dataset..."
    img_train, mask_train, filenames_train = load_images(
        img_train_path, gt_train_path, camvid_colors, load_greylevel_mask=True,
        save=False)  # load_greylevel_mask=True by default because it's grey

    print "Processing Camvid SegNet valid dataset..."
    img_valid, mask_valid, filenames_valid = load_images(
        img_valid_path, gt_valid_path, camvid_colors, load_greylevel_mask=True,
        save=False)  # load_greylevel_mask=True by default because it's grey

    print "Processing Camvid SegNet test dataset..."
    img_test, mask_test, filenames_test = load_images(
        img_test_path, gt_test_path, camvid_colors, load_greylevel_mask=True,
        save=False)  # load_greylevel_mask=True by default because it's grey

    return (img_train, mask_train, filenames_train,
            img_test, mask_test, filenames_test,
            img_valid, mask_valid, filenames_valid)


def load_data(
    path=os.path.expanduser('~/exp/datasets/camvid/'),
    randomize=False,
    resize_images=True,
    resize_size=[320, 240],  # w x h : 960x720, 480x360, 320x240
    color=False,
    color_space='RGB',
    normalize=False,
    classes='subset_11',  # subset_11 , all
    version='segnet',  # standard, segnet
    split=[.44, .22],
    with_filenames=False,
    load_greylevel_mask=False,
    save=False,
    compute_stats='all',
    rng=None,
    with_fullmasks=False
):
    """Dataset loader

    Parameter
    ---------
    path : string the path to the dataset images.
    randomize               False
    resize                  False
    use_fullsize_images     True
    version: string
        standard, segnet
    compute_stas: string
        train, all
    """
    #############
    # LOAD DATA #
    #############

    if version == 'segnet':
        path = os.path.join(path, 'segnet')
        (img_train,
         mask_train,
         filenames_train,
         img_test,
         mask_test,
         filenames_test,
         img_val,
         mask_val,
         filenames_val) = load_dataset_camvid_segnet(path)

    elif version == 'standard':
        path = os.path.join(path, 'splitted_960x720')
        (img_train,
         mask_train,
         filenames_train,
         img_test,
         mask_test,
         filenames_test,
         img_val,
         mask_val,
         filenames_val) = load_dataset_camvid(path,
                                              resize_images=resize_images,
                                              resize_size=resize_size,
                                              load_greylevel_mask=
                                              load_greylevel_mask,
                                              classes=classes,
                                              save=save,
                                              color_space=color_space
                                              )

    if compute_stats == 'all':
        images = np.asarray(img_train + img_val + img_test)
    elif compute_stats == 'train':
        images = np.asarray(img_train)

    # all images have the same dimension --> we can compute per pixel statistics
    mean = images.mean(axis=0)[np.newaxis, ...]
    std = np.maximum(images.std(axis=0), 1e-8)[np.newaxis, ...]
    print "Computing dataset statistics ..."

    # split datasets
    ntrain = len(img_train)
    ntest = len(img_test)
    nvalid = len(img_val)
    ntot = ntrain + ntest + nvalid

    train_set_x = np.array(img_train)
    train_set_y = np.array(mask_train)
    test_set_x = np.array(img_test)
    test_set_y = np.array(mask_test)
    valid_set_x = np.array(img_val)
    valid_set_y = np.array(mask_val)

    u_train, c_train = np.unique(train_set_y, return_counts=True)
    u_valid, c_valid = np.unique(valid_set_y, return_counts=True)
    u_test, c_test = np.unique(test_set_y, return_counts=True)

    print u_train
    print np.round(100 * c_train / np.sum(c_train), 2)

    print u_valid
    print np.round(100 * c_valid / np.sum(c_valid), 2)

    print u_test
    print np.round(100 * c_test / np.sum(c_test), 2)

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    filenames = [np.array(filenames_train),
                 np.array(filenames_val),
                 np.array(filenames_test)]
    print "load_data Done!"
    print('Tot images:{} Train:{} Valid:{} Test:{}').format(
        ntot, ntrain, nvalid, ntest)

    """
    # Debug for types
    print (train_set_x.dtype)
    print (test_set_x.dtype)
    print (valid_set_x.dtype)


    print (train_set_y.dtype)
    print (test_set_y.dtype)
    print (valid_set_y.dtype)

    print (train_set_x[0].dtype)
    print (test_set_x[0].dtype)
    print (valid_set_x[0].dtype)


    print (train_set_y[0].dtype)
    print (test_set_y[0].dtype)
    print (valid_set_y[0].dtype)
    """

    out_list = [train, valid, test, mean, std]
    if with_filenames:
        out_list.append(filenames)
    if with_fullmasks:
        out_list.append([])

    return out_list

if __name__ == '__main__':
    load_data(save=False)
