import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty


def buildReSeg(
        input_shape,
        n_layers,
        pheight,
        pwidth,
        dim_proj,
        nclasses,
        stack_sublayers,
        out_upsampling,
        out_nfilters,
        out_filters_size,
        out_filters_stride):

    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    weights_loss = T.scalar('weights_loss')

    # INPUT LAYER it has the input dimension:
    # canvas dimension:
    # batch_size, heigth, width, channels
    batch_size, cheight, cwidth, cchannels = input_shape
    l_in = lasagne.layers.InputLayer(shape=input_shape,
                                     input_var=input_var,
                                     name="input_layer"
                                     )

    # TODO insert input conv layers
    # Here I introduce some inputs valid convolutional layers
    # An idea could be to use VGG16 structure (maybe also the pretrained
    # weights) and then ReNet with 1x1 patch.
    # The I can use InverseLayer to invert the input conv layers
    # for now I add only some input convolution in order to extract some
    # features
    # and use less memory for the rnns

    for idx_layers in xrange(n_layers):
        # number of blocks in each direction
        nblocksH = cheight / pheight[idx_layers]
        nblocksW = cwidth / pwidth[idx_layers]
        # I PREPARE THE CANVAS INPUT TO GO IN THE NEXT RENET LAYER
        # I divide the image in blocks of the dimension of the patch_size
        # So I reorder in such a way that each block is flattened over the
        # channel dimension
        # so we have:
        # batch_size, n_block_height, n_block_width, n_el_each_block*channels
        l_in = lasagne.layers.ReshapeLayer(
                l_in,
                (batch_size,
                 nblocksH,
                 pheight[idx_layers],
                 nblocksW,
                 pwidth[idx_layers],
                 cchannels),
                name="reshape_0_"+str(idx_layers))
        l_in = lasagne.layers.DimshuffleLayer(
                l_in, (0, 1, 3, 2, 4, 5),
                name="dimshuffle_0_"+str(idx_layers))
        l_in = lasagne.layers.ReshapeLayer(
                l_in, (batch_size,
                       nblocksH,
                       nblocksW,
                       pheight[idx_layers] * pwidth[idx_layers] * cchannels),
                name="reshape_1_"+str(idx_layers))

        # RENET LAYER
        renet_layer_out, out_shape_layer = buildReNetLayer(
                l_in,
                (batch_size, nblocksH, nblocksW, cchannels),
                patch=(pwidth[idx_layers], pheight[idx_layers]),
                n_hidden=dim_proj[idx_layers],
                stack_sublayers=stack_sublayers[idx_layers],
                n_layer=idx_layers)

        # TODO: insert dimensional reduction 1x1 Conv layer after ReNet layer

        _, cheight,  cwidth, cchannels = out_shape_layer
        l_in = renet_layer_out

        n_rnns = 2 if stack_sublayers[idx_layers] else 4
        print('ReNet: After {} rnns {}x{} @ {}: {}'.format(
                n_rnns,
                pheight[idx_layers],
                pwidth[idx_layers],
                dim_proj[idx_layers],
                out_shape_layer))

    # UPSAMPLING
    if out_upsampling == 'grad':
        # We use a custom Deconv layer: UpsampleConv2DDNNLayer
        # the input have have to be in the 'bc01' shape
        renet_layer_out = lasagne.layers.dimshuffle(renet_layer_out,
                                                    (0, 3, 1, 2))

        for i, (num_filters, filter_size, filter_stride) in enumerate(zip(
                out_nfilters, out_filters_size, out_filters_stride)):
            renet_layer_out = UpsampleConv2DDNNLayer(
                    renet_layer_out,
                    num_filters=num_filters,
                    filter_size=filter_size,
                    stride=filter_stride,
                    pad='same')

        out_layer = lasagne.layers.DimshuffleLayer(renet_layer_out,
                                                   (0, 2, 3, 1))

    elif out_upsampling == 'linear':
        expand_height = np.prod(pheight)
        expand_width = np.prod(pwidth)
        out_layer = LinearUpsamplingLayer(renet_layer_out,
                                          expand_height,
                                          expand_width,
                                          nclasses,
                                          name="linear_upsample_layer")

    # Reshape in 2D, last dimension is nclasses, where the softmax is applied
    out_layer = lasagne.layers.ReshapeLayer(out_layer,
                                            [-1, out_nfilters[-1]],
                                            name='reshape_before_softmax')
    out_layer = lasagne.layers.NonlinearityLayer(
            out_layer,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="softmax_layer")

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):

    prediction = lasagne.layers.get_output(out_layer)

    loss = lasagne.objectives.categorical_crossentropy(
            prediction, target_var)
    loss = weights_loss * loss.mean()

    # TODO We could add some weight decay as well here,
    # see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(out_layer, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=LEARNING_RATE, momentum=0.9)
    updates = lasagne.updates.adadelta(loss, params)

    test_prediction = lasagne.layers.get_output(out_layer, deterministic=True)
    # this is the function that gives back the mask prediction
    f_pred = theano.function([input_var],
                             T.argmax(test_prediction, axis=1).reshape(
                                     input_shape[:3]))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    f_train = theano.function([input_var, target_var, weights_loss], loss,
                              updates=updates)

    return out_layer, f_pred, f_train


def buildReNetLayer(input,
                    shape,
                    patch=(2, 2),
                    n_hidden=50,
                    stack_sublayers=False,
                    **kwargs):
    """
    Each ReNet layer contains 4 rnns:
            First SubLayer:
                2 rnns scans the image vertically (up and down)
            Second Sublayer:
                2 rnns scans the ima

        The sublayers can be stack_sublayers or can scan in parallel the image

    :param input:
    :param shape:
    :param patch:
    :param n_hidden:
    :param stack_sublayers:
    :param kwargs:
    :return:
    """

    n_layer = kwargs.get("n_layer", 0)

    # canvas dimension
    batch_size, nblocksH, nblocksW, cchannels = shape
    # patch dimension
    pheight, pwidth = patch

    # actually the real dimensions now are:
    # batch_size, nblocksH, nblocksW, pheight * pwidth * cchannels
    # it means if the original image was:
    # batch_size, 240, 320, 3
    # and patch_size = (2, 2)
    # then the dimensions are
    # batch_size, 120, 160, 3*2*2=12

    # FIRST SUBLAYER
    # The GRU Layer needs a 3D tensor input
    shape = (batch_size * nblocksH, nblocksW, pheight * pwidth * cchannels)
    reshape_input_sublayer1 = lasagne.layers.ReshapeLayer(
            input,
            shape,
            name="renet_reshape_in_"+str(n_layer))

    # I want to iterate over each block so I swap
    reshape_input_sublayer1 = lasagne.layers.DimshuffleLayer(
            reshape_input_sublayer1,
            (1, 0, 2))

    # left/right scan
    sub_layer1_out = buildReNetSublayer(
            reshape_input_sublayer1, n_hidden,
            suffix=str(n_layer) + "_" + str(0))

    # after the bidirectional rnns now I have the following 3D shape:
    # nblocksW, batch_size * nblocksH, 2*n_hidden
    # so I revert the swap and restore the 4D tensor
    shape_after_sublayer1 = (batch_size, nblocksH, nblocksW, 2*n_hidden)

    sub_layer1_out = lasagne.layers.DimshuffleLayer(
            sub_layer1_out,
            (1, 0, 2))
    reshape_after_sublayer1 = lasagne.layers.ReshapeLayer(
        sub_layer1_out,
        shape_after_sublayer1,
        name="renet_reshape_0_after_"+str(n_layer))

    # The second sublayer can be stacked or not
    # If is stack_sublayers it takes in input the first sublayer output,
    # otherwise the input is the same of the first sublayer (e.g the image)
    if stack_sublayers:
        input_sublayer2 = reshape_after_sublayer1
        cchannels = 2 * n_hidden
    else:
        input_sublayer2 = input
        cchannels = cchannels * pwidth * pheight

    # SECOND SUBLAYER
    # vertical/horizontal scan

    reshape_input_sublayer2 = lasagne.layers.DimshuffleLayer(
            input_sublayer2,
            (0, 2, 1, 3),
            name="renet_reshape_1_after_"+str(n_layer))

    reshape_input_sublayer2 = lasagne.layers.ReshapeLayer(
            reshape_input_sublayer2,
            (batch_size * nblocksW, nblocksH, cchannels),
            name="renet_reshape_2_after_"+str(n_layer))
    # I want to iterate over each block so I swap
    reshape_input_sublayer2 = lasagne.layers.DimshuffleLayer(
            reshape_input_sublayer2,
            (1, 0, 2))

    # down/up
    sub_layer2_out = buildReNetSublayer(reshape_input_sublayer2,
                                        n_hidden,
                                        suffix=str(n_layer) + "_" + str(1))

    # after the bidirectional rnns now I have the following 3D shape:
    # nblocksH, batch_size * nblocksW, 2*n_hidden
    # so I revert the swap and restore the 4D tensor
    shape_after_sublayer2 = (batch_size, nblocksW, nblocksH, 2 * n_hidden)
    reshape_after_sublayer2 = lasagne.layers.DimshuffleLayer(
            sub_layer2_out,
            (1, 0, 2))

    reshape_after_sublayer2 = lasagne.layers.ReshapeLayer(
            reshape_after_sublayer2,
            shape_after_sublayer2,
            name="renet_reshape_3_after_"+str(n_layer))

    reshape_after_sublayer2 = lasagne.layers.DimshuffleLayer(
            reshape_after_sublayer2,
            (0, 2, 1, 3),
            name="renet_reshape_4_after_"+str(n_layer))

    if not stack_sublayers:
        output = lasagne.layers.ConcatLayer(
                [reshape_after_sublayer1, reshape_after_sublayer2],
                axis=3)
        output_shape = (batch_size, nblocksH, nblocksW, 4 * n_hidden)
    else:
        output = reshape_after_sublayer2
        output_shape = (batch_size, nblocksH, nblocksW, 2 * n_hidden)

    return output, output_shape


def buildReNetSublayer(incoming,
                       num_units,
                       # resetgate=lasagne.layers.Gate(
                       #     W_in=lasagne.init.Orthogonal(1.0),
                       #     W_hid=lasagne.init.Orthogonal(1.0),
                       #     W_cell=lasagne.init.Orthogonal(1.0),
                       #     b=lasagne.init.Constant(0.),
                       #     nonlinearity=lasagne.nonlinearities.tanh),
                       # updategate=lasagne.layers.Gate(
                       #     W_in=lasagne.init.Orthogonal(1.0),
                       #     W_hid=lasagne.init.Orthogonal(1.0),
                       #     W_cell=lasagne.init.Orthogonal(1.0),
                       #     b=lasagne.init.Constant(0.),
                       #     nonlinearity=lasagne.nonlinearities.tanh),
                       # hidden_update=lasagne.layers.Gate(
                       #     W_in=lasagne.init.Orthogonal(1.0),
                       #     W_hid=lasagne.init.Orthogonal(1.0),
                       #     W_cell=lasagne.init.Orthogonal(1.0),
                       #     b=lasagne.init.Constant(0.),
                       #     nonlinearity=lasagne.nonlinearities.tanh),
                       # hid_init=lasagne.init.Constant(0.),
                       grad_clipping=10,
                       **kwargs):

    suffix = kwargs.get("suffix", "0_0")

    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    # Setting only_return_final=True makes the layers only return their
    # output for the final time step, which is all we need for this task
    l_forward = lasagne.layers.GRULayer(
        incoming,
        num_units,
        # resetgate=resetgate,
        # updategate=updategate,
        # hidden_update=hidden_update,
        # hid_init=hid_init,
        # grad_clipping=grad_clipping,
        only_return_final=False,
        name='l_forward_sub_' + suffix)
    l_backward = lasagne.layers.GRULayer(
        l_forward,
        num_units,
        # resetgate=resetgate,
        # updategate=updategate,
        # hidden_update=hidden_update,
        # hid_init=hid_init,
        # grad_clipping=grad_clipping,
        only_return_final=False,
        backwards=True,
        name='l_backward_sub_' + suffix)
    # Now, we'll concatenate the outputs to combine them.
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2,)

    return l_concat


class LinearUpsamplingLayer(lasagne.layers.Layer):

    def __init__(self, incoming,
                 expand_height,
                 expand_width,
                 nclasses,
                 W=lasagne.init.Normal(0.01),
                 b=lasagne.init.Constant(.0),
                 **kwargs):
        super(LinearUpsamplingLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        num_units = expand_height * expand_width * nclasses

        self.num_units = num_units
        self.expand_height = expand_height
        self.expand_width = expand_width
        self.nclasses = nclasses
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.b = self.add_param(b, (num_units,), name='b')

    def get_output_for(self, input, **kwargs):
        # upsample
        pred = T.dot(input, self.W) + self.b

        batch_size, nrows, ncolumns, _ = self.input_shape
        nclasses = self.nclasses
        expand_height = self.expand_height
        expand_width = self.expand_width

        # Reshape after the upsampling to come back to the original
        # dimensions and move the pixels in the right place
        pred = pred.reshape((batch_size,
                             nrows,
                             ncolumns,
                             expand_height,
                             expand_width,
                             nclasses))
        pred = pred.dimshuffle((0, 1, 4, 2, 5, 3))
        pred = pred.reshape((batch_size,
                             nrows * expand_height,
                             ncolumns * expand_width,
                             nclasses))
        return pred

    def get_output_shape_for(self, input_shape):
        batch_size, nrows, ncolumns, _ = input_shape
        return (input_shape[0],
                input_shape[1] * self.expand_height,
                input_shape[2] * self.expand_width,
                self.nclasses)


def deconv_length(output_length, filter_size, stride, pad=0):
    if output_length is None:
        return None

    output_length = output_length * stride
    if pad == 'valid':
        input_length = output_length + filter_size - 1
    elif pad == 'full':
        input_length = output_length - filter_size + 1
    elif pad == 'same':
        input_length = output_length
    elif isinstance(pad, int):
        input_length = output_length - 2 * pad + filter_size - 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    return input_length


class DeconvLayer(lasagne.layers.Layer):
    """
    This is a Deconvolutional layer, basically the same that was implemented
    in ReSeg framework, but now is a Lasagne layer.
    It's an adaptation of 'ebenolson' implementation
    with  some improvement using cuda sandox functions
    https://github.com/ebenolson/Lasagne/blob/deconv/lasagne/layers/dnn.py

    Note: remeber to pass the input in the form of
            batch, cchannel, row, col
    """

    def __init__(self, incoming, num_filters, filter_size, stride=(2, 2),
                 pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 flip_filters=False, **kwargs):
        super(DeconvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2)
        self.stride = lasagne.utils.as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad == 'full':
            self.pad = 'full'
        elif pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
            self.pad = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        else:
            self.pad = lasagne.utils.as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2],
                                self.output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = deconv_length(input_shape[2],
                                    self.filter_size[0],
                                    self.stride[0],
                                    pad[0])

        output_columns = deconv_length(input_shape[3],
                                       self.filter_size[1],
                                       self.stride[1],
                                       pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'

        filters = gpu_contiguous(self.W)
        state_below = gpu_contiguous(input)
        out_shape = T.stack(self.output_shape)

        desc = GpuDnnConvDesc(border_mode=self.pad,
                              subsample=self.stride,
                              conv_mode=conv_mode)(out_shape, filters.shape)
        grad = GpuDnnConvGradI()(filters, state_below,
                                 gpu_alloc_empty(*out_shape), desc)

        # image = T.alloc(0., *self.output_shape)
        # conved = dnn_conv(img=image,
        #                   kerns=self.W,
        #                   subsample=self.stride,
        #                   border_mode=self.pad,
        #                   conv_mode=conv_mode
        #                   )
        #
        # grad = T.grad(conved.sum(), wrt=image, known_grads={conved: input})

        if self.b is None:
            activation = grad
        elif self.untie_biases:
            activation = grad + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = grad + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)


class UpsampleConv2DDNNLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=(2, 2),
                 pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 flip_filters=False, **kwargs):
        super(UpsampleConv2DDNNLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2)
        self.stride = lasagne.utils.as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad == 'full':
            self.pad = 'full'
        elif pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
            self.pad = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        else:
            self.pad = lasagne.utils.as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2],
                                self.output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = deconv_length(input_shape[2],
                                    self.filter_size[0],
                                    self.stride[0],
                                    pad[0])

        output_columns = deconv_length(input_shape[3],
                                       self.filter_size[1],
                                       self.stride[1],
                                       pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'

        image = T.alloc(0., *self.output_shape)
        conved = dnn_conv(img=image,
                          kerns=self.W,
                          subsample=self.stride,
                          border_mode=self.pad,
                          conv_mode=conv_mode
                          )

        grad = T.grad(conved.sum(), wrt=image, known_grads={conved: input})

        if self.b is None:
            activation = grad
        elif self.untie_biases:
            activation = grad + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = grad + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)