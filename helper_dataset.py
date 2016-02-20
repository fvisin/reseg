import numpy as np
import os

from numpy import sqrt, prod, ones, floor, repeat, pi, exp, zeros, sum
from numpy.random import RandomState

from theano.tensor.nnet import conv2d
from theano import shared, config, _asarray, function
import theano.tensor as T
floatX = config.floatX

from sklearn.feature_extraction.image import PatchExtractor
from sklearn.decomposition import PCA

from skimage import exposure
from skimage import io
from skimage import img_as_float, img_as_ubyte, img_as_uint, img_as_int
from skimage.color import label2rgb, rgb2hsv, hsv2rgb
from skimage.io import ImageCollection, imsave
from skimage.transform import resize


def compare_mask_image_filenames(filenames_images, filenames_mask,
                                 replace_from='',
                                 replace_to='',
                                 msg="Filename images and mask mismatch"):
    image = [i.split('/')[-1] for i in filenames_images]
    mask = [i.split('/')[-1].replace(replace_from, replace_to) for i in
            filenames_mask]

    assert np.array_equal(image, mask), msg


def convert_RGB_mask_to_index(im, colors, ignore_missing_labels=False):
    """
    :param im: mask in RGB format (classes are RGB colors)
    :param colors: the color map should be in the following format

         colors = OrderedDict([
            ("Sky", np.array([[128, 128, 128]], dtype=np.uint8)),
            ("Building", np.array([[128, 0, 0],   # Building
                               [64, 192, 0],  # Wall
                               [0, 128, 64]   # Bridge
                               ], dtype=np.uint8)
            ...
                               ])

    :param ignore_missing_labels: if True the function continue also if some
    pixels fail the mappint
    :return: the mask in index class format
    """

    out = (np.ones(im.shape[:2]) * 255).astype(np.uint8)
    for grey_val, (label, rgb) in enumerate(colors.items()):
        for el in rgb:
            match_pxls = np.where((im == np.asarray(el)).sum(-1) == 3)
            out[match_pxls] = grey_val
            if ignore_missing_labels:  # retrieve the void label
                if [0, 0, 0] in rgb:
                    void_label = grey_val
    # debug
    # outpath = '/Users/marcus/exp/datasets/camvid/grey_test/o.png'
    # imsave(outpath, out)
    ######

    if ignore_missing_labels:
        match_missing = np.where(out == 255)
        if match_missing[0].size > 0:
            print "Ignoring missing labels"
            out[match_missing] = void_label

    assert (out != 255).all(), "rounding errors or missing classes in colors"
    return out.astype(np.uint8)


def resize():
    pass


def crop():
    pass


def zero_pad(im, resize_size, inpath="", pad_value=0):
    """

    :param im: the image you want to resize
    :param resize_size: the new size of the image
    :param inpath: [optional] to debug, the path of the image
    :return: the zero-pad image in the new dimensions
    """
    if im.ndim == 3:
        h, w, _ = im.shape
    elif im.ndim == 2:
        h, w = im.shape

    rw, rh = resize_size

    pad_w = rw - w
    pad_h = rh - h

    pad_l = pad_r = pad_u = pad_d = 0
    if pad_w > 0:
        pad_l = int(pad_w / 2)
        pad_r = pad_w - pad_l

    if pad_h > 0:
        pad_u = int(pad_h / 2)
        pad_d = pad_h - pad_u

    if im.ndim == 3:
        im = np.pad(im, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)),
                    mode='constant',
                    constant_values=pad_value)
    elif im.ndim == 2:
        im = np.pad(im, ((pad_u, pad_d), (pad_l, pad_r)),
                    mode='constant',
                    constant_values=pad_value)

    assert (im.shape[1], im.shape[0]) == resize_size, \
        "Resize size doesn't match: resize_size->{} resized->{}"\
        " filename : {}".format(resize_size,
                                [im.shape[1], im.shape[0]],
                                inpath
                                )
    return im


def rgb2illumination_invariant(img, alpha, hist_eq=False):
    """
    this is an implementation of the illuminant-invariant color space published
    by Maddern2014
    http://www.robots.ox.ac.uk/~mobile/Papers/2014ICRA_maddern.pdf

    :param img:
    :param alpha: camera paramete
    :return:
    """
    ii_img = 0.5 + np.log(img[:, :, 1] + 1e-8) - \
        alpha * np.log(img[:, :, 2] + 1e-8) - \
        (1 - alpha) * np.log(img[:, :, 0] + 1e-8)

    # ii_img = exposure.rescale_intensity(ii_img, out_range=(0, 1))
    if hist_eq:
        ii_img = exposure.equalize_hist(ii_img)

    print np.max(ii_img)
    print np.min(ii_img)

    return ii_img


def save_image(outpath, img):
    import errno
    try:
        os.makedirs(os.path.dirname(outpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass
    imsave(outpath, img)


def save_RGB_mask(outpath, mask):
    return


def preprocess_dataset(train, valid, test,
                       preprocess_type,
                       patch_size, max_patches):

    if preprocess_type is None:
        return train, valid, test

    # whiten, LCN, GCN, Local Mean Subtract, or normalize
    print("Preprocessing train set with ", preprocess_type, " ", patch_size)
    train_pre = []
    for x in train[0]:
        # TODO: add an axis before everything
        x_pre = preprocess(np.expand_dims(x, axis=0), preprocess_type,
                           patch_size,
                           max_patches)
        train_pre.append(x_pre[0])
    train = (np.array(train_pre), np.array(train[1]))

    print("Preprocessing valid set with ", preprocess_type, " ", patch_size)
    valid_pre = []
    for x in valid[0]:
        x_pre = preprocess(np.expand_dims(x, axis=0), preprocess_type,
                           patch_size,
                           max_patches)
        valid_pre.append(x_pre[0])
    valid = (np.array(valid_pre), np.array(valid[1]))

    print("Preprocessing test set with ", preprocess_type, " ", patch_size)
    test_pre = []
    for x in test[0]:
        x_pre = preprocess(np.expand_dims(x, axis=0), preprocess_type,
                           patch_size,
                           max_patches)
        test_pre.append(x_pre[0])
    test = (np.array(test_pre), np.array(test[1]))

    return train, valid, test


def preprocess(x, mode=None,
               patch_size=9,
               max_patches=int(1e5)):
    """

    :param x:
    :param mode:
    :param rng:
    :param patch_size:
    :param max_patches:
    :return:
    """

    if mode == 'conv-zca':
        x = convolutional_zca(x,
                              patch_size=patch_size,
                              max_patches=max_patches)
    elif mode == 'sub-lcn':
        for d in range(x.shape[-1]):
            x[:, :, :, d] = lecun_lcn(x[:, :, :, d],
                                      kernel_size=patch_size)
    elif mode == 'subdiv-lcn':
        for d in range(x.shape[-1]):
            x[:, :, :, d] = lecun_lcn(x[:, :, :, d],
                                      kernel_size=patch_size,
                                      use_divisor=True)
    elif mode == 'gcn':
        for d in range(x.shape[-1]):
            x[:, :, :, d] = global_contrast_normalization(x[:, :, :, d])
    elif mode == 'local_mean_sub':
        for d in range(x.shape[-1]):
            x[:, :, :, d] = local_mean_subtraction(x[:, :, :, d],
                                                   kernel_size=patch_size)
    x = x.astype(floatX)
    return x


def lecun_lcn(input, kernel_size=9, threshold=1e-4, use_divisor=False):
    """
    Yann LeCun's local contrast normalization
    Orginal code in Theano by: Guillaume Desjardins

    :param input:
    :param kernel_size:
    :param threshold:
    :param use_divisor:
    :return:
    """
    input_shape = (input.shape[0], 1, input.shape[1], input.shape[2])
    input = input.reshape(input_shape).astype(floatX)

    X = T.tensor4(dtype=floatX)
    filter_shape = (1, 1, kernel_size, kernel_size)
    filters = gaussian_filter(kernel_size).reshape(filter_shape)
    filters = shared(_asarray(filters, dtype=floatX), borrow=True)

    convout = conv2d(input=X,
                     filters=filters,
                     input_shape=input.shape,
                     filter_shape=filter_shape,
                     border_mode='full')

    # For each pixel, remove mean of kernel_size x kernel_size neighborhood
    mid = int(floor(kernel_size/2.))
    new_X = X - convout[:, :, mid:-mid, mid:-mid] # this is centered

    if use_divisor:
        # Scale down norm of kernel_size x kernel_size patch
        sum_sqr_XX = conv2d(input=T.sqr(T.abs_(new_X)),
                            filters=filters,
                            input_shape=input.shape,
                            filter_shape=filter_shape,
                            border_mode='full')

        denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
        per_img_mean = denom.mean(axis=[2, 3])
        divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
        divisor = T.maximum(divisor, threshold)
        new_X = new_X / divisor

    new_X = new_X.dimshuffle(0, 2, 3, 1)
    new_X = new_X.flatten(ndim=3)
    f = function([X], new_X)
    return f(input)


def local_mean_subtraction(input, kernel_size=5):

    input_shape = (input.shape[0], 1, input.shape[1], input.shape[2])
    input = input.reshape(input_shape).astype(floatX)

    X = T.tensor4(dtype=floatX)
    filter_shape = (1, 1, kernel_size, kernel_size)
    filters = mean_filter(kernel_size).reshape(filter_shape)
    filters = shared(_asarray(filters, dtype=floatX), borrow=True)

    mean = conv2d(input=X,
                  filters=filters,
                  input_shape=input.shape,
                  filter_shape=filter_shape,
                  border_mode='full')
    mid = int(floor(kernel_size/2.))

    new_X = X - mean[:, :, mid:-mid, mid:-mid]
    f = function([X], new_X)
    return f(input)


def global_contrast_normalization(input, scale=1., subtract_mean=True,
    use_std=False, sqrt_bias=0., min_divisor=1e-8):

    input_shape = (input.shape[0], 1, input.shape[1], input.shape[2])
    input = input.reshape(input_shape).astype(floatX)

    X = T.tensor4(dtype=floatX)
    ndim = X.ndim
    if not ndim in [3, 4]:
        raise NotImplementedError("X.dim>4 or X.ndim<3")

    scale = float(scale)
    mean = X.mean(axis=ndim-1)
    new_X = X.copy()

    if subtract_mean:
        if ndim == 3:
            new_X = X - mean[:, :, None]
        else:
            new_X = X - mean[:, :, :, None]

    if use_std:
        normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim-1)) / scale
    else:
        normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim-1)) / scale

    # Don't normalize by anything too small.
    T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

    if ndim == 3:
        new_X /= normalizers[:, :, None]
    else:
        new_X /= normalizers[:, :, :, None]

    f = function([X], new_X)
    return f(input)


def gaussian_filter(kernel_shape):

    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma**2
        return 1./Z * exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i, j] = gauss(i-mid, j-mid)

    return x / sum(x)


def mean_filter(kernel_size):
    s = kernel_size**2
    x = repeat(1. / s, s).reshape((kernel_size, kernel_size))
    return x


def convolutional_zca(input, patch_size=(9, 9), max_patches=int(1e5)):
    """
    This is an implementation of the convolutional ZCA whitening presented by
    David Eigen in his phd thesis
    http://www.cs.nyu.edu/~deigen/deigen-thesis.pdf

    "Predicting Images using Convolutional Networks:
     Visual Scene Understanding with Pixel Maps"

    From paragraph 8.4:
    A simple adaptation of ZCA to convolutional application is to find the
    ZCA whitening transformation for a sample of local image patches across
    the dataset, and then apply this transform to every patch in a larger image.
    We then use the center pixel of each ZCA patch to create the conv-ZCA
    output image. The operations of applying local ZCA and selecting the center
    pixel can be combined into a single convolution kernel,
    resulting in the following algorithm
    (explained using RGB inputs and 9x9 kernel):

    1. Sample 10M random 9x9 image patches (each with 3 colors)
    2. Perform PCA on these to get eigenvectors V and eigenvalues D.
    3. Optionally remove small eigenvalues, so V has shape [npca x 3 x 9 x 9].
    4. Construct the whitening kernel k:
        for each pair of colors (ci,cj),
        set k[j,i, :, :] = V[:, j, x0, y0]^T * D^{-1/2} * V[:, i, :, :]

    where (x0, y0) is the center pixel location (e.g. (5,5) for a 9x9 kernel)


    :param input: 4D tensor of shape [batch_size, rows, col, channels]
    :param patch_size: size of the patches extracted from the dataset
    :param max_patches: max number of patches extracted from the dataset

    :return: conv-zca whitened dataset
    """
    patch_size = (patch_size, patch_size)
    # I don't know if it's correct or not.. but it seems to work
    mean = np.mean(input, axis=(0, 1, 2))
    input -= mean  # center the data

    n_imgs, h, w, n_channels = input.shape
    patches = PatchExtractor(patch_size=patch_size,
                             max_patches=max_patches).transform(input)
    pca = PCA()
    pca.fit(patches.reshape(patches.shape[0], -1))

    # Transpose the components into theano convolution filter type
    dim = (-1,) + patch_size + (n_channels,)
    V = shared(pca.components_.reshape(dim).
               transpose(0, 3, 1, 2).astype(input.dtype))
    D = T.nlinalg.diag(1. / np.sqrt(pca.explained_variance_))

    x_0 = int(np.floor(patch_size[0] / 2))
    y_0 = int(np.floor(patch_size[1] / 2))

    filter_shape = [n_channels, n_channels, patch_size[0], patch_size[1]]
    image_shape = [n_imgs, n_channels, h, w]
    kernel = T.zeros(filter_shape)
    VT = V.dimshuffle(2, 3, 1, 0)

    # V : 243 x 3 x 9 x 9
    # VT : 9 x 9 x 3 x 243

    # build the kernel
    for i in range(n_channels):
        for j in range(n_channels):
            a = T.dot(VT[x_0, y_0, j, :], D).reshape([1, -1])
            b = V[:, i, :, :].reshape([-1, patch_size[0] * patch_size[1]])
            c = T.dot(a, b).reshape([patch_size[0], patch_size[1]])
            kernel = T.set_subtensor(kernel[j, i, :, :], c)

    kernel = kernel.astype(floatX)
    input = input.astype(floatX)
    input_images = T.tensor4(dtype=floatX)
    conv_whitening = conv2d(input_images.dimshuffle((0, 3, 1, 2)),
                            kernel,
                            input_shape=image_shape,
                            filter_shape=filter_shape,
                            border_mode='full'
                            )

    s_crop = [(patch_size[0] - 1) // 2,
              (patch_size[1] - 1) // 2]
    # e_crop = [s_crop[0] if (s_crop[0] % 2) != 0 else s_crop[0] + 1,
    #           s_crop[1] if (s_crop[1] % 2) != 0 else s_crop[1] + 1]

    conv_whitening = conv_whitening[:, :, s_crop[0]:-s_crop[0], s_crop[
        1]:-s_crop[1]]
    conv_whitening = conv_whitening.dimshuffle(0, 2, 3, 1)
    f_convZCA = function([input_images], conv_whitening)

    return f_convZCA(input)