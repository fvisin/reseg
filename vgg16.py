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
from lasagne.layers import (InputLayer, DenseLayer,
                            NonlinearityLayer, ConcatLayer)
from lasagne.nonlinearities import softmax
from lasagne.layers import get_output_shape, get_output
from padded import PaddedConv2DLayer
from padded import PaddedPool2DLayer
import theano


class Vgg16Layer(lasagne.layers.Layer):
    def __init__(self,
                 l_in=InputLayer((None, 3, 224, 224)),
                 get_layer='prob',
                 use_concat_layers=False,
                 get_concat_layer='concat3',
                 padded=True,
                 trainable=False, regularizable=False,
                 name='vgg'):

        super(Vgg16Layer, self).__init__(l_in, name)
        self.l_in = l_in
        self.get_layer = get_layer
        self.use_concat_layers = use_concat_layers
        self.get_concat_layer = get_concat_layer
        self.padded = padded
        self.trainable = trainable
        self.regularizable = regularizable

        if padded:
            ConvLayer = PaddedConv2DLayer
            PoolLayer = PaddedPool2DLayer
        else:
            try:
                ConvLayer = lasagne.layers.dnn.Conv2DDNNLayer
            except AttributeError:
                ConvLayer = lasagne.layers.Conv2DLayer
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
            net['fc8'] = DenseLayer(net['fc7'],
                                    num_units=1000,
                                    nonlinearity=None)
            net['prob'] = NonlinearityLayer(net['fc8'], softmax)

        if use_concat_layers:
            # concatenation of different feature maps
            concat_net = OrderedDict()
            concat_net['input'] = net['bgr']

            concat_net['conv1'] = ConvLayer(
                concat_net['input'], 64, 2, pad=0, stride=2,
                flip_filters=False)

            concat_net['pool1'] = net['pool1']
            concat_net['concat1'] = ConcatLayer(
                [concat_net['conv1'], concat_net['pool1']], axis=1,
                name='concat1')

            concat_net['conv2'] = ConvLayer(
                concat_net['concat1'], 128, 2, pad=0, stride=2,
                flip_filters=False)

            concat_net['pool2'] = net['pool2']
            concat_net['concat2'] = ConcatLayer(
                [concat_net['conv2'], concat_net['pool2']], axis=1,
                name='concat2')

            concat_net['conv3'] = ConvLayer(
                concat_net['concat2'], 256, 2, pad=0, stride=2,
                flip_filters=False)

            concat_net['pool3'] = net['pool3']
            concat_net['concat3'] = ConcatLayer(
                [concat_net['conv3'], concat_net['pool3']], axis=1,
                name='concat3')

            concat_net['conv4'] = ConvLayer(
                concat_net['concat3'], 512, 2, pad=0, stride=2,
                flip_filters=False)
            concat_net['pool4'] = net['pool4']
            concat_net['concat4'] = ConcatLayer(
                [concat_net['conv4'], concat_net['pool4']], axis=1,
                name='concat4')

            reached = False
            # Collect garbage
            for el in concat_net.iteritems():
                if reached:
                    del(concat_net[el[0]])
                if el[0] == get_concat_layer:
                    reached = True

            # Set names to layers
            for name in concat_net.keys():
                concat_net[name].name = 'vgg16_merge_' + name

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

        if use_concat_layers:
            self.out_layer = concat_net[get_concat_layer]
            self.sublayers = concat_net
        else:
            self.out_layer = net[get_layer]
            self.sublayers = net

         # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(self.out_layer, tuple):
            self.input_layer = None
        else:
            self.input_layer = self.out_layer

    def get_output_shape_for(self, input_shape):
        for name, layer in self.sublayers.items():
            if not self.use_concat_layers and 'input' in name:
                continue
            if self.use_concat_layers and 'pool' in name:
                continue
            if self.use_concat_layers and 'concat' in name:
                input_shape = (input_shape, input_shape)

            output_shape = layer.get_output_shape_for(input_shape)
            input_shape = output_shape
        return output_shape

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


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
