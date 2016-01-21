import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import get_output_shape
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty


def buildReSeg(input_shape, n_layers, pheight, pwidth, dim_proj, nclasses,
               stack_sublayers, out_upsampling, out_nfilters, out_filters_size,
               out_filters_stride):
    '''Helper function to build a ReSeg network

    '''
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    weights_loss = T.scalar('weights_loss')

    # Tag test values
    input_var.tag.test_value = np.random.random(
        input_shape).astype('float32')
    target_var.tag.test_value = np.random.random(
        list(input_shape[:3]) + [nclasses]).astype('float32')
    # theano.config.compute_test_value = 'warn'

    # The ReSeg layer
    print('Input shape: ' + str(input_shape))
    l_in = lasagne.layers.InputLayer(shape=input_shape,
                                     input_var=input_var,
                                     name="input_layer")

    l_reseg = ReSegLayer(l_in, n_layers, pheight, pwidth, dim_proj,
                         nclasses, stack_sublayers, out_upsampling,
                         out_nfilters, out_filters_size,
                         out_filters_stride, 'reseg')

    # Reshape in 2D, last dimension is nclasses, where the softmax is applied
    l_out = lasagne.layers.ReshapeLayer(
        l_reseg,
        (T.prod(l_reseg.output_shape[0:3]), [3]),
        name='reshape_before_softmax')

    l_pred = lasagne.layers.NonlinearityLayer(
            l_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="softmax_layer")

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(l_pred)

    loss = lasagne.objectives.categorical_crossentropy(
        prediction, target_var)
    loss = weights_loss * loss.mean()

    # TODO We could add some weight decay as well here,
    # see lasagne.regularization.

    params = lasagne.layers.get_all_params(l_pred, trainable=True)
    # Stochastic Gradient Descent (SGD) with Nesterov momentum
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=LEARNING_RATE, momentum=0.9)
    updates = lasagne.updates.adadelta(loss, params)

    # Compile the function that gives back the mask prediction
    prediction = lasagne.layers.get_output(l_pred, deterministic=True)
    f_pred = theano.function(
        [input_var],
        T.argmax(prediction, axis=1).reshape(input_shape[:3]))

    # Compile the function that performs a training step on a mini-batch
    # (by using the updates dictionary) and returns the corresponding training
    # loss:
    f_train = theano.function([input_var, target_var, weights_loss], loss,
                              updates=updates)

    return l_out, f_pred, f_train


class ReSegLayer(lasagne.layers.Layer):
    def __init__(self, l_in, n_layers, pheight, pwidth, dim_proj, nclasses,
                 stack_sublayers, out_upsampling, out_nfilters,
                 out_filters_size, out_filters_stride, name=''):
        """A ReSeg layer

        The ReSeg layer is composed by multiple ReNet layers and an
        upsampling layer

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer
        n_layers : int
            The number of layers
        pheight : tuple
            The height of the patches, for each layer
        pwidth : tuple
            The width of the patches, for each layer
        dim_proj : tuple
            The number of hidden units of each RNN, for each layer
        nclasses : int
            The number of classes of the data
        stack_sublayers : bool
            If True the bidirectional RNNs in the ReNet layers will be
            stacked one over the other. See ReNet for more details.
        out_upsampling : string
            The kind of upsampling to be used
        out_nfilters : int
            The number of hidden units of the upsampling layer
        out_filters_size : tuple
            The size of the upsampling filters, if any
        out_filters_stride : tuple
            The stride of the upsampling filters, if any
        name : string
            The name of the layer, optional
        """

        super(ReSegLayer, self).__init__(l_in, name)
        self.l_in = l_in
        self.n_layers = n_layers
        self.pheight = pheight
        self.pwidth = pwidth
        self.dim_proj = dim_proj
        self.nclasses = nclasses
        self.stack_sublayers = stack_sublayers
        self.out_upsampling = out_upsampling
        self.out_nfilters = out_nfilters
        self.out_filters_size = out_filters_size
        self.out_filters_stride = out_filters_stride
        self.name = name

        (batch_size, cheight, cwidth, cchannels) = get_output_shape(l_in)

        # TODO insert input conv layers

        # ReNet layers
        l_renet = l_in
        for lidx in xrange(n_layers):
            l_renet = ReNetLayer(l_renet, patch=(pwidth[lidx], pheight[lidx]),
                                 n_hidden=dim_proj[lidx],
                                 stack_sublayers=stack_sublayers[lidx],
                                 name=self.name + '_renet' + str(lidx))
            out_shape = get_output_shape(l_renet)

            # TODO: insert dimensional reduction 1x1 Conv layer after ReNet

            n_rnns = 2 if stack_sublayers[lidx] else 4
            print('ReNet: After {} rnns {}x{} @ {}: {}'.format(
                n_rnns, pheight[lidx], pwidth[lidx], dim_proj[lidx],
                out_shape))

        # UPSAMPLING
        # TODO We could use InverseLayer to invert the input conv layers (?)
        if out_upsampling == 'grad':
            # We use a custom Deconv layer: UpsampleConv2DDNNLayer
            # the input has to be in the 'bc01' shape
            l_renet_out = lasagne.layers.dimshuffle(l_renet,
                                                    (0, 3, 1, 2))

            for i, (num_filters, filter_size, filter_stride) in enumerate(zip(
                    out_nfilters, out_filters_size, out_filters_stride)):
                renet_layer_out = UpsampleConv2DDNNLayer(
                        l_renet_out,
                        num_filters=num_filters,
                        filter_size=filter_size,
                        stride=filter_stride,
                        pad='same')

            l_out = lasagne.layers.DimshuffleLayer(renet_layer_out,
                                                   (0, 2, 3, 1))

        elif out_upsampling == 'linear':
            expand_height = np.prod(pheight)
            expand_width = np.prod(pwidth)
            l_out = LinearUpsamplingLayer(l_renet,
                                          expand_height,
                                          expand_width,
                                          nclasses,
                                          name="linear_upsample_layer")
        self.l_out = l_out

        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(self.l_out, tuple):
            self.input_layer = None
        else:
            self.input_layer = self.l_out

    def get_output_shape_for(self, input_shape):
        return list(input_shape[0:3]) + [self.nclasses]

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


class ReNetLayer(lasagne.layers.Layer):

    def __init__(self, l_in, patch_size=(2, 2), n_hidden=50,
                 stack_sublayers=False, name='', **kwargs):
        """A ReNet layer

        Each ReNet layer is composed by 4 RNNs (or 2 bidirectional RNNs):
        * First SubLayer:
            2 RNNs scan the image vertically (up and down)
        * Second Sublayer:
            2 RNNs scan the image horizontally (left and right)

        The sublayers can be stacked one over the other or can scan the
        image in parallel

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer
        patch_size : tuple
            The size of the patch expressed as (pheight, pwidth).
            Optional
        n_hidden : int
            The number of hidden units of each RNN. Optional
        stack_sublayers : bool
            If True, the sublayers (i.e. the bidirectional RNNs) will be
            stacked one over the other, meaning that the second
            bidirectional RNN will read the feature map coming from the
            first bidirectional RNN. If False, all the RNNs will read
            the input. Optional
        name : string
            The name of the layer, optional
        """
        super(ReNetLayer, self).__init__(l_in, name)
        self.l_in = l_in
        self.patch_size = patch_size
        self.n_hidden = n_hidden
        self.stack_sublayers = stack_sublayers
        self.name = name

        batch_size, cheight, cwidth, cchannels = get_output_shape(l_in)
        pheight, pwidth = patch_size

        # Number of patches in each direction
        npatchesH = cheight / pheight
        npatchesW = cwidth / pwidth

        # Split in patches
        l_sub0 = lasagne.layers.ReshapeLayer(
            l_in,
            (batch_size, npatchesH, pheight, npatchesW, pwidth, cchannels),
            name=self.name + "_reshape0")

        l_sub0 = lasagne.layers.DimshuffleLayer(
            l_sub0,
            (0, 1, 3, 2, 4, 5),
            name=self.name + "_dimshuffle0")

        l_sub0 = lasagne.layers.ReshapeLayer(
            l_sub0,
            (batch_size, npatchesH, npatchesW, pheight * pwidth * cchannels),
            name=self.name + "_reshape1")

        # FIRST SUBLAYER
        # The GRU Layer needs a 3D tensor input
        l_sub0 = lasagne.layers.ReshapeLayer(
            l_sub0,
            (batch_size * npatchesH, npatchesW, pheight * pwidth * cchannels),
            name=self.name + "_sub0_reshape")

        # Iterate over columns
        l_sub0 = lasagne.layers.DimshuffleLayer(
            l_sub0,
            (1, 0, 2),
            name=self.name + "_sub0_dimshuffle")

        # Left/right scan
        l_sub0 = BidirectionalRNNLayer(
            l_sub0,
            n_hidden,
            name=self.name + "_sub0_renetsub")

        # Revert dimshuffle
        l_sub0 = lasagne.layers.DimshuffleLayer(
            l_sub0,
            (1, 0, 2),
            name=self.name + "_sub0_undimshuffle")

        # Revert reshape
        l_sub0 = lasagne.layers.ReshapeLayer(
            l_sub0,
            (batch_size, npatchesH, npatchesW, 2 * n_hidden),
            name=self.name + "_sub0_unreshape")

        # If stack_sublayers is True, the second sublayer takes as an input the
        # first sublayer's output, otherwise the input of the ReNetLayer (e.g
        # the image)
        if stack_sublayers:
            input_sublayer1 = l_sub0
            cchannels = 2 * n_hidden
        else:
            input_sublayer1 = l_in
            # cchannels = cchannels * pwidth * pheight

        # SECOND SUBLAYER
        # Invert rows and columns
        l_sub1 = lasagne.layers.DimshuffleLayer(
            input_sublayer1,
            (0, 2, 1, 3),
            name=self.name + "_sub1_dimshuffle0")

        # The GRU Layer needs a 3D tensor input
        l_sub1 = lasagne.layers.ReshapeLayer(
            l_sub1,
            (batch_size * npatchesW, npatchesH, cchannels),
            name=self.name + "_sub1_reshape")

        # Iterate over rows
        l_sub1 = lasagne.layers.DimshuffleLayer(
            l_sub1,
            (1, 0, 2),
            name=self.name + "_sub1_dimshuffle1")

        # Down/up scan
        l_sub1 = BidirectionalRNNLayer(
            l_sub1,
            n_hidden,
            name=self.name + "_sub1_renetsub")

        # Revert the last dimshuffle
        l_sub1 = lasagne.layers.DimshuffleLayer(
            l_sub1,
            (1, 0, 2),
            name=self.name + "_sub1_undimshuffle1")

        # Revert the reshape
        l_sub1 = lasagne.layers.ReshapeLayer(
            l_sub1,
            (batch_size, npatchesW, npatchesH, 2 * n_hidden),
            name=self.name + "_sub1_unreshape")

        # Revert the other dimshuffle
        l_sub1 = lasagne.layers.DimshuffleLayer(
            l_sub1,
            (0, 2, 1, 3),
            name=self.name + "_sub1_undimshuffle0")

        # Set out_layer and out_shape
        if not stack_sublayers:
            self.out_layer = lasagne.layers.ConcatLayer(
                [l_sub0, l_sub1],
                axis=3)
        else:
            self.out_layer = l_sub1

        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(self.out_layer, tuple):
            self.input_layer = None
        else:
            self.input_layer = self.out_layer

    def get_output_shape_for(self, input_shape):
        pheight, pwidth = self.patch_size
        npatchesH = input_shape[1] / pheight
        npatchesW = input_shape[2] / pwidth

        if self.stack_sublayers:
            dim = 2 * self.n_hidden
        else:
            dim = 4 * self.n_hidden

        return input_shape[0], npatchesH, npatchesW, dim

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


class BidirectionalRNNLayer(lasagne.layers.Layer):

    # Setting a value for grad_clipping will clip the gradients in the layer
    def __init__(self, l_in, num_units,
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
                 grad_clipping=0, name='', **kwargs):
        """BidirectionalRNNLayer

        A bidirectional RNN.

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer
        num_units : int
            The number of hidden units of each RNN
        grad_clipping : int
            The amount of gradient clipping, optional
        name = string
            The name of the layer, optional
        """
        super(BidirectionalRNNLayer, self).__init__(l_in, name, **kwargs)
        self.l_in = l_in
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.name = name

        # We use a bidirectional RNN, which means we combine two
        # RecurrentLayers, the second of which with backwards=True
        # Setting only_return_final=True makes the layers only return their
        # output for the final time step, which is all we need for this task
        l_forward = lasagne.layers.GRULayer(
            l_in,
            num_units,
            # resetgate=resetgate,
            # updategate=updategate,
            # hidden_update=hidden_update,
            # hid_init=hid_init,
            grad_clipping=grad_clipping,
            only_return_final=False,
            name=name + '_l_forward_sub')
        l_backward = lasagne.layers.GRULayer(
            l_forward,
            num_units,
            # resetgate=resetgate,
            # updategate=updategate,
            # hidden_update=hidden_update,
            # hid_init=hid_init,
            grad_clipping=grad_clipping,
            only_return_final=False,
            backwards=True,
            name=name + '_l_backward_sub')

        # Now we'll concatenate the outputs to combine them
        # Note that l_backward is already inverted by Lasagne
        l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward],
                                              axis=2, name=name+'_concat')

        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(l_concat, tuple):
            self.input_layer = None
        else:
            self.input_layer = l_concat

    def get_output_shape_for(self, input_shape):
        return list(input_shape[0:2]) + [self.num_units * 2]

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


class LinearUpsamplingLayer(lasagne.layers.Layer):

    def __init__(self,
                 incoming,
                 expand_height,
                 expand_width,
                 nclasses,
                 W=lasagne.init.Normal(0.01),
                 b=lasagne.init.Constant(.0),
                 **kwargs):
        super(LinearUpsamplingLayer, self).__init__(incoming, **kwargs)
        nfeatures_in = self.input_shape[-1]
        nfeatures_out = expand_height * expand_width * nclasses

        self.nfeatures_out = nfeatures_out
        self.incoming = incoming
        self.expand_height = expand_height
        self.expand_width = expand_width
        self.nclasses = nclasses
        self.W = self.add_param(W, (nfeatures_in, nfeatures_out), name='W')
        self.b = self.add_param(b, (nfeatures_out,), name='b')

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
