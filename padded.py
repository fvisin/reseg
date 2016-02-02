import warnings

from lasagne import init, nonlinearities
from lasagne.layers.dnn import Conv2DDNNLayer
from theano import tensor as T
from theano.ifelse import ifelse


class PaddedConv2DLayer(Conv2DDNNLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=True, convolution=T.nnet.conv2d, **kwargs):
        if pad != 0:
            warnings.warn('The specified padding will be ignored',
                          RuntimeWarning)
        super(PaddedConv2DLayer, self).__init__(incoming, num_filters,
                                                filter_size, stride, pad,
                                                untie_biases, W, b,
                                                nonlinearity, flip_filters,
                                                **kwargs)
        if self.input_shape[2:] != [None, None]:
            raise(NotImplementedError('This Layer should only be used when '
                                      'the size of the image is not known'))

    def convolve(self, input_im, **kwargs):
        # Compute the padding required not to crop any pixel
        in_shape = input_im.shape[2:]  # bs, ch, 0, 1
        in_shape -= self.W.shape[2:]
        stride = self.stride
        pad = in_shape % stride
        pad = (stride - pad) % stride
        # Zero pad
        input_im = ifelse(
            T.eq(pad[0], 0),
            input_im,
            T.concatenate((T.zeros_like(input_im[:, :, 0:pad[0], :]),
                           input_im), 2))
        input_im = ifelse(
            T.eq(pad[1], 0),
            input_im,
            T.concatenate((T.zeros_like(input_im[:, :, :, 0:pad[1]]),
                           input_im), 3))
        # Erase self.pad to prevent theano from padding the input
        self.pad = 0
        ret = super(PaddedConv2DLayer, self).convolve(input_im, **kwargs)
        # Set pad to access it from outside
        self.pad = pad
        return ret
