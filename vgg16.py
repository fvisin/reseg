# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from collections import OrderedDict
import numpy
try:
    import cPickle as pickle
except:
    import pickle

import lasagne
import lasagne.layers
import lasagne.layers.dnn
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax
from padded import PaddedConv2DLayer
from padded import PaddedPool2DLayer
import theano


def buildVgg16(l_in=InputLayer((None, 3, 224, 224)), get_layer='prob',
               padded=True, trainable=False, regularizable=False):
    if padded:
        ConvLayer = PaddedConv2DLayer
        PoolLayer = PaddedPool2DLayer
    else:
        ConvLayer = lasagne.layers.dnn.Conv2DDNNLayer
        PoolLayer = lasagne.layers.Pool2DLayer

    net = OrderedDict()
    net['input'] = l_in
    net['bgr'] = RGBtoBGRLayer(net['input'])
    net['conv1_1'] = ConvLayer(
        net['bgr'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(
        net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(
        net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(
        net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(
        net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    if 'fc' in get_layer or get_layer == 'prob':
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
        net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    reached = False
    # Collect garbage
    for el in net.iteritems():
        if reached:
            del(net[el[0]])
        if el[0] == get_layer:
            reached = True

    # Set names to layers
    for name in net.keys():
        net[name].name = 'vgg16_' + name

    # Reload weights
    nparams = len(lasagne.layers.get_all_params(net.values()))
    with open('w_vgg16.pkl', 'rb') as f:
        # Note: in python3 use the pickle.load parameter `encoding='latin-1'`
        vgg16_w = pickle.load(f)['param values']
    lasagne.layers.set_all_param_values(net.values(), vgg16_w[:nparams])

    # Do not train or regularize vgg
    if not trainable or not regularizable:
        all_layers = net.values()
        for vgg_layer in all_layers:
            layer_params = vgg_layer.get_params()
            for p in layer_params:
                if not regularizable:
                    try:
                        vgg_layer.params[p].remove('regularizable')
                    except KeyError:
                        pass
                if not trainable:
                    try:
                        vgg_layer.params[p].remove('trainable')
                    except KeyError:
                        pass

    return net[get_layer]


class RGBtoBGRLayer(lasagne.layers.Layer):
    def __init__(self, l_in, bgr_mean=numpy.array([103.939, 116.779, 123.68]),
                 data_format='bc01', **kwargs):
        """A Layer to normalize and convert images from RGB to BGR

        This layer converts images from RGB to BGR to adapt to Caffe
        that uses OpenCV, which uses BGR. It also subtracts the
        per-pixel mean.

        Parameters
        ----------
        l_in : :class:``lasagne.layers.Layer``
            The incoming layer, typically an
            :class:``lasagne.layers.InputLayer``
        bgr_mean : iterable of 3 ints
            The mean of each channel. By default, the ImageNet
            mean values are used.
        data_format : str
            The format of l_in, either `b01c` (batch, rows, cols,
            channels) or `bc01` (batch, channels, rows, cols)
        """
        super(RGBtoBGRLayer, self).__init__(l_in, **kwargs)
        assert data_format in ['bc01', 'b01c']
        self.l_in = l_in
        floatX = theano.config.floatX
        self.bgr_mean = bgr_mean.astype(floatX)
        self.data_format = data_format

    def get_output_for(self, input_im, **kwargs):
        if self.data_format == 'bc01':
            input_im = input_im[:, ::-1, :, :]
            input_im -= self.bgr_mean[:, numpy.newaxis, numpy.newaxis]
        else:
            input_im = input_im[:, :, :, ::-1]
            input_im -= self.bgr_mean
        return input_im
