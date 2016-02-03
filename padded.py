import warnings

from lasagne import init, nonlinearities
from lasagne.layers import Pool2DLayer, Conv2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer
import theano
from theano import tensor as T
from theano.ifelse import ifelse


class PaddedConv2DLayer(Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=True, convolution=theano.tensor.nnet.conv2d,
                 **kwargs):
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
        **kwargs
            Any additional keyword arguments are passed to the
            :class:``lasagne.layers.Layer`` superclass
        """
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
        in_shape = input_arr.shape[2:]  # bs, ch, rows, cols
        in_shape -= self.W.shape[2:]
        stride = self.stride
        pad = in_shape % stride
        pad = (stride - pad) % stride
        # Zero pad
        input_arr = ifelse(
            T.eq(pad[0], 0),
            input_arr,
            T.concatenate((T.zeros_like(input_arr[:, :, 0:pad[0], :]),
                           input_arr), 2))
        input_arr = ifelse(
            T.eq(pad[1], 0),
            input_arr,
            T.concatenate((T.zeros_like(input_arr[:, :, :, 0:pad[1]]),
                           input_arr), 3))
        # Erase self.pad to prevent theano from padding the input
        self.pad = 0
        ret = super(PaddedConv2DLayer, self).get_output_for(input_arr,
                                                            **kwargs)
        # Set pad to access it from outside
        self.pad = pad
        return ret

    def get_equivalent_input_padding(self, layers_args=[]):
        """Compute the equivalent padding in the input layer

        See :func:`padded.get_equivalent_input_padding`
        """
        return(get_equivalent_input_padding(self, layers_args))


class PaddedPool2DLayer(Pool2DLayer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, **kwargs):
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
        **kwargs
            Any additional keyword arguments are passed to the Layer
            superclass
        """
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
        in_shape = input_arr.shape[2:]  # bs, ch, rows, cols
        in_shape -= self.W.shape[2:]
        stride = self.stride
        pad = in_shape % stride
        pad = (stride - pad) % stride
        # Zero pad
        input_arr = ifelse(
            T.eq(pad[0], 0),
            input_arr,
            T.concatenate((T.zeros_like(input_arr[:, :, 0:pad[0], :]),
                           input_arr), 2))
        input_arr = ifelse(
            T.eq(pad[1], 0),
            input_arr,
            T.concatenate((T.zeros_like(input_arr[:, :, :, 0:pad[1]]),
                           input_arr), 3))
        # Erase self.pad to prevent theano from padding the input
        self.pad = 0
        ret = super(PaddedConv2DLayer, self).convolve(input_arr, **kwargs)
        # Set pad to access it from outside
        self.pad = pad
        return ret

    def get_equivalent_input_padding(self, layers_args=[]):
        """Compute the equivalent padding in the input layer

        See :func:`padded.get_equivalent_input_padding`
        """
        return(get_equivalent_input_padding(self, layers_args))


def get_equivalent_input_padding(layer, layers_args=[]):
    """Compute the equivalent padding in the input layer

    A function to compute the equivalent padding of a sequence of
    convolutional and pooling layers. It memorizes the padding
    of all the Layers up to the first Layer that is not an instance of
    :class:``lasagne.layers.Pool2DLayer``,
    :class:``lasagne.layers.Conv2DLayer``, or
    :class:``lasagne.layers.Conv2DDNNLayer`` (which includes the padded
    variants defined here). It then computes what would be the equivalent
    padding in the Layer immediately before the chain of Layers that is
    being taken into account.
    """
    # Loop through conv and pool to collect data
    while(isinstance(layer, (Pool2DLayer, Conv2DLayer, Conv2DDNNLayer))):
        # Note: stride is numerical, but pad *could* be symbolic
        pad, stride = (layer.pad, layer.stride)
        if isinstance(pad, int):
            pad = pad, pad
        if isinstance(stride, int):
            stride = stride, stride
        layers_args.append((pad, stride))
        layer = layer.input_layer

    # Loop backward to compute the equivalent padding in the input
    # layer
    tot_pad = T.zeros(2)
    pad_factor = T.ones(2)
    while(layers_args):
        pad, stride = layers_args.pop()
        tot_pad += pad * pad_factor
        pad_factor *= stride

    return tot_pad
