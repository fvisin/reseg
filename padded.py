import warnings

import numpy
import lasagne
from lasagne import init, nonlinearities
from lasagne.layers import get_all_layers, Conv2DLayer, Layer, Pool2DLayer
import theano
from theano import tensor as T
from theano.ifelse import ifelse


class PaddedConv2DLayer(Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=True, convolution=theano.tensor.nnet.conv2d,
                 centered=True, **kwargs):
        """A padded convolutional layer

        Note
        ----
        If used in place of a :class:``lasagne.layers.Conv2DLayer`` be
        sure to specify `flag_filters=False`, which is the default for
        that layer

        Parameters
        ----------
        incoming : lasagne.layers.Layer
            The input layer
        num_filters : int
            The number of filters or kernels of the convolution
        filter_size : int or iterable of int
            The size of the filters
        stride : int or iterable of int
            The stride or subsampling of the convolution
        pad :  int, iterable of int, ``full``, ``same`` or ``valid``
            **Ignored!** Kept for compatibility with the
            :class:``lasagne.layers.Conv2DLayer``
        untie_biases : bool
            See :class:``lasagne.layers.Conv2DLayer``
        W : Theano shared variable, expression, numpy array or callable
            See :class:``lasagne.layers.Conv2DLayer``
        b : Theano shared variable, expression, numpy array, callable or None
            See :class:``lasagne.layers.Conv2DLayer``
        nonlinearity : callable or None
            See :class:``lasagne.layers.Conv2DLayer``
        flip_filters : bool
            See :class:``lasagne.layers.Conv2DLayer``
        convolution : callable
            See :class:``lasagne.layers.Conv2DLayer``
        centered : bool
            If True, the padding will be added on both sides. If False
            the zero padding will be applied on the upper left side.
        **kwargs
            Any additional keyword arguments are passed to the
            :class:``lasagne.layers.Layer`` superclass
        """
        self.centered = centered
        if pad not in [0, (0, 0), [0, 0]]:
            warnings.warn('The specified padding will be ignored',
                          RuntimeWarning)
        super(PaddedConv2DLayer, self).__init__(incoming, num_filters,
                                                filter_size, stride, pad,
                                                untie_biases, W, b,
                                                nonlinearity, flip_filters,
                                                **kwargs)
        if self.input_shape[2:] != (None, None):
            warnings.warn('This Layer should only be used when the size of '
                          'the image is not known', RuntimeWarning)

    def get_output_for(self, input_arr, **kwargs):
        # Compute the padding required not to crop any pixel
        input_arr, pad = zero_pad(
            input_arr, self.filter_size, self.stride, self.centered, 'bc01')

        # Erase self.pad to prevent theano from padding the input
        self.pad = 0
        ret = super(PaddedConv2DLayer, self).get_output_for(input_arr,
                                                            **kwargs)
        # Set pad to access it from outside
        self.pad = pad
        return ret

    def get_output_shape_for(self, input_shape):
        return zero_pad_shape(input_shape, self.filter_size, self.stride,
                              'bc01')

    def get_equivalent_input_padding(self, layers_args=[]):
        """Compute the equivalent padding in the input layer

        See :func:`padded.get_equivalent_input_padding`
        """
        return(get_equivalent_input_padding(self, layers_args))


class PaddedPool2DLayer(Pool2DLayer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, centered=True, **kwargs):
        """A padded pooling layer

        Parameters
        ----------
        incoming : lasagne.layers.Layer
            The input layer
        pool_size : int
            The size of the pooling
        stride : int or iterable of int
            The stride or subsampling of the convolution
        pad :  int, iterable of int, ``full``, ``same`` or ``valid``
            **Ignored!** Kept for compatibility with the
            :class:``lasagne.layers.Pool2DLayer``
        ignore_border : bool
            See :class:``lasagne.layers.Pool2DLayer``
        centered : bool
            If True, the padding will be added on both sides. If False
            the zero padding will be applied on the upper left side.
        **kwargs
            Any additional keyword arguments are passed to the Layer
            superclass
        """
        self.centered = centered
        if pad not in [0, (0, 0), [0, 0]]:
            warnings.warn('The specified padding will be ignored',
                          RuntimeWarning)
        super(PaddedPool2DLayer, self).__init__(incoming,
                                                pool_size,
                                                stride,
                                                pad,
                                                ignore_border,
                                                **kwargs)
        if self.input_shape[2:] != (None, None):
            warnings.warn('This Layer should only be used when the size of '
                          'the image is not known', RuntimeWarning)

    def get_output_for(self, input_arr, **kwargs):
        # Compute the padding required not to crop any pixel
        input_arr, pad = zero_pad(
            input_arr, self.pool_size, self.stride, self.centered, 'bc01')
        # Erase self.pad to prevent theano from padding the input
        self.pad = 0
        ret = super(PaddedConv2DLayer, self).convolve(input_arr, **kwargs)
        # Set pad to access it from outside
        self.pad = pad
        return ret

    def get_output_shape_for(self, input_shape):
        return zero_pad_shape(input_shape, self.pool_size, self.stride,
                              'bc01')

    def get_equivalent_input_padding(self, layers_args=[]):
        """Compute the equivalent padding in the input layer

        See :func:`padded.get_equivalent_input_padding`
        """
        return(get_equivalent_input_padding(self, layers_args))


class DynamicPaddingLayer(Layer):
    def __init__(
            self,
            l_in,
            patch_size,
            stride,
            data_format='bc01',
            centered=True,
            name='',
            **kwargs):
        """A Layer that zero-pads the input

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer
        patch_size :  iterable of int
            The patch size
        stride : iterable of int
            The stride
        data_format : string
            The format of l_in, either `b01c` (batch, rows, cols,
            channels) or `bc01` (batch, channels, rows, cols)
        centered : bool
            If True, the padding will be added on both sides. If False
            the zero padding will be applied on the upper left side.
        name = string
            The name of the layer, optional
        """
        super(DynamicPaddingLayer, self).__init__(l_in, name, **kwargs)
        self.l_in = l_in
        self.patch_size = patch_size
        self.stride = stride
        self.data_format = data_format
        self.centered = centered
        self.name = name

    def get_output_for(self, input_arr, **kwargs):
        input_arr, pad = zero_pad(
            input_arr, self.patch_size, self.stride, self.centered,
            self.data_format)
        self.pad = pad
        return input_arr

    def get_output_shape_for(self, input_shape):
        return zero_pad_shape(input_shape, self.patch_size, self.stride,
                              self.data_format, True)


def zero_pad(input_arr, patch_size, stride, centered=True, data_format='bc01'):
    assert data_format in ['bc01', 'b01c']

    if data_format == 'b01c':
        in_shape = input_arr.shape[1:3]
    else:
        in_shape = input_arr.shape[2:]  # bs, ch, rows, cols
    in_shape -= patch_size
    pad = in_shape % stride
    pad = (stride - pad) % stride

    # TODO improve efficiency by allocating the full array of zeros and
    # setting the subtensor afterwards
    if data_format == 'bc01':
        if centered:
            input_arr = ifelse(
                T.eq(pad[0], 0),
                input_arr,
                T.concatenate(
                    (T.zeros_like(input_arr[:, :, :pad[0]/2, :]),
                     input_arr,
                     T.zeros_like(input_arr[:, :, :pad[0] - pad[0]/2, :])),
                    2))
            input_arr = ifelse(
                T.eq(pad[1], 0),
                input_arr,
                T.concatenate(
                    (T.zeros_like(input_arr[:, :, :, :pad[1]/2]),
                     input_arr,
                     T.zeros_like(input_arr[:, :, :, :pad[1] - pad[1]/2])),
                    3))
        else:
            input_arr = ifelse(
                T.eq(pad[0], 0),
                input_arr,
                T.concatenate((T.zeros_like(input_arr[:, :, :pad[0], :]),
                               input_arr), 2))
            input_arr = ifelse(
                T.eq(pad[1], 0),
                input_arr,
                T.concatenate((T.zeros_like(input_arr[:, :, :, :pad[1]]),
                               input_arr), 3))
    else:
        if centered:
            input_arr = ifelse(
                T.eq(pad[0], 0),
                input_arr,
                T.concatenate(
                    (T.zeros_like(input_arr[:, :pad[0]/2, :, :]),
                     input_arr,
                     T.zeros_like(input_arr[:, :pad[0] - pad[0]/2, :, :])),
                    1))
            input_arr = ifelse(
                T.eq(pad[1], 0),
                input_arr,
                T.concatenate(
                    (T.zeros_like(input_arr[:, :, :pad[1]/2, :]),
                     input_arr,
                     T.zeros_like(input_arr[:, :, :pad[1] - pad[1]/2, :])),
                    2))
        else:
            input_arr = ifelse(
                T.eq(pad[0], 0),
                input_arr,
                T.concatenate((T.zeros_like(input_arr[:, :pad[0], :, :]),
                               input_arr), 1))
            input_arr = ifelse(
                T.eq(pad[1], 0),
                input_arr,
                T.concatenate((T.zeros_like(input_arr[:, :, :pad[1], :]),
                               input_arr), 2))
    return input_arr, pad


def zero_pad_shape(input_shape, patch_size, stride, data_format,
                   only_pad=False):
    assert data_format in ['bc01', 'b01c']
    patch_size = numpy.array(patch_size)
    stride = numpy.array(stride)

    if data_format == 'b01c':
        im_shape = numpy.array(input_shape[1:3])
    else:
        im_shape = numpy.array(input_shape[2:])
    pad = (im_shape - patch_size) % stride
    pad = (stride - pad) % stride

    if only_pad:
        out_shape = list(im_shape + pad)
    else:
        out_shape = list((im_shape - patch_size + pad) / stride + 1)

    if data_format == 'b01c':
        out_shape = [input_shape[0]] + out_shape + [input_shape[3]]
    else:
        out_shape = list(input_shape[:2]) + out_shape
    return list(out_shape)


def get_equivalent_input_padding(layer, layers_args=[]):
    """Compute the equivalent padding in the input layer

    A function to compute the equivalent padding of a sequence of
    convolutional and pooling layers. It memorizes the padding
    of all the Layers up to the first InputLayer.
    It then computes what would be the equivalent padding in the Layer
    immediately before the chain of Layers that is being taken into account.
    """
    # Initialize the DynamicPadding layers
    lasagne.layers.get_output(layer)
    # Loop through conv and pool to collect data
    all_layers = get_all_layers(layer)
    # while(not isinstance(layer, (InputLayer))):
    for layer in all_layers:
        # Note: stride is numerical, but pad *could* be symbolic
        try:
            pad, stride = (layer.pad, layer.stride)
            if isinstance(pad, int):
                pad = pad, pad
            if isinstance(stride, int):
                stride = stride, stride
            layers_args.append((pad, stride))
        except(AttributeError):
            pass

    # Loop backward to compute the equivalent padding in the input
    # layer
    tot_pad = T.zeros(2)
    pad_factor = T.ones(2)
    while(layers_args):
        pad, stride = layers_args.pop()
        tot_pad += pad * pad_factor
        pad_factor *= stride

    return tot_pad
