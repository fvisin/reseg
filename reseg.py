# Standard library imports
import cPickle as pkl
import collections
import os
import random
from shutil import move, rmtree
import sys
import time

# Related third party imports
import lasagne
from lasagne.layers import get_output
import numpy as np
from progressbar import ProgressBar
import theano
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode

# Local application/library specific imports
from helper_dataset import preprocess_dataset
from get_info_model import print_params
from layers import CropLayer, ReSegLayer
from subprocess import check_output
from utils import iterate_minibatches, save_with_retry, validate, VariableText

# Datasets import
# TODO these should go into preprocess/helper dataset/evaluate
import camvid
import daimler
import fashionista
import flowers
import horses
import kitti_road
import mscoco
import nyu_depth
import sunrgbd

floatX = theano.config.floatX
intX = 'uint8'

debug = False
nanguard = False

datasets = {'camvid': (camvid.load_data, camvid.properties),
            'daimler': (daimler.load_data, daimler.properties),
            'fashionista': (fashionista.load_data, fashionista.properties),
            'flowers': (flowers.load_data, flowers.properties),
            'horses': (horses.load_data, horses.properties),
            'kitti_road': (kitti_road.load_data, kitti_road.properties),
            'mscoco': (mscoco.load_data, mscoco.properties),
            'nyu_depth': (nyu_depth.load_data, nyu_depth.properties),
            'sunrgbd': (sunrgbd.load_data, sunrgbd.properties)}


def get_dataset(name):
    return (datasets[name][0], datasets[name][1])


def buildReSeg(input_shape, input_var,
               n_layers, pheight, pwidth, dim_proj,
               nclasses, stack_sublayers,
               # upsampling
               out_upsampling,
               out_nfilters,
               out_filters_size,
               out_filters_stride,
               out_W_init=lasagne.init.GlorotUniform(),
               out_b_init=lasagne.init.Constant(0.),
               out_nonlinearity=lasagne.nonlinearities.rectify,
               # input ConvLayers
               in_nfilters=None,
               in_filters_size=(),
               in_filters_stride=(),
               in_W_init=lasagne.init.GlorotUniform(),
               in_b_init=lasagne.init.Constant(0.),
               in_nonlinearity=lasagne.nonlinearities.rectify,
               # common recurrent layer params
               RecurrentNet=lasagne.layers.GRULayer,
               nonlinearity=lasagne.nonlinearities.rectify,
               hid_init=lasagne.init.Constant(0.),
               grad_clipping=0,
               precompute_input=True,
               mask_input=None,
               # 1x1 Conv layer for dimensional reduction
               conv_dim_red=False,
               conv_dim_red_nonlinearity=lasagne.nonlinearities.identity,
               # GRU specific params
               gru_resetgate=lasagne.layers.Gate(W_cell=None),
               gru_updategate=lasagne.layers.Gate(W_cell=None),
               gru_hidden_update=lasagne.layers.Gate(
                   W_cell=None,
                   nonlinearity=lasagne.nonlinearities.tanh),
               gru_hid_init=lasagne.init.Constant(0.),
               # LSTM specific params
               lstm_ingate=lasagne.layers.Gate(),
               lstm_forgetgate=lasagne.layers.Gate(),
               lstm_cell=lasagne.layers.Gate(
                   W_cell=None,
                   nonlinearity=lasagne.nonlinearities.tanh),
               lstm_outgate=lasagne.layers.Gate(),
               # RNN specific params
               rnn_W_in_to_hid=lasagne.init.Uniform(),
               rnn_W_hid_to_hid=lasagne.init.Uniform(),
               rnn_b=lasagne.init.Constant(0.),
               # Special layer
               batch_norm=False
               ):
    '''Helper function to build a ReSeg network'''

    # Input is b01c
    print('Input shape: ' + str(input_shape))
    l_in = lasagne.layers.InputLayer(shape=input_shape,
                                     input_var=input_var,
                                     name="input_layer")

    # Convert to bc01 (batchsize, ch, rows, cols)
    l_in = lasagne.layers.DimshuffleLayer(l_in, (0, 3, 1, 2))

    # To know the upsampling ratio we compute what is the feature map
    # size at the end of the downsampling pathway for an hypotetical
    # initial size of 100 (we just need the ratio, so we don't care
    # about the actual size)
    hypotetical_fm_size = np.array((100.0, 100.0))

    l_reseg = ReSegLayer(l_in, n_layers, pheight, pwidth, dim_proj,
                         nclasses, stack_sublayers,
                         # upsampling
                         out_upsampling,
                         out_nfilters,
                         out_filters_size,
                         out_filters_stride,
                         out_W_init=out_W_init,
                         out_b_init=out_b_init,
                         out_nonlinearity=out_nonlinearity,
                         hypotetical_fm_size=hypotetical_fm_size,
                         # input ConvLayers
                         in_nfilters=in_nfilters,
                         in_filters_size=in_filters_size,
                         in_filters_stride=in_filters_stride,
                         in_W_init=in_W_init,
                         in_b_init=in_b_init,
                         in_nonlinearity=in_nonlinearity,
                         # common recurrent layer params
                         RecurrentNet=RecurrentNet,
                         nonlinearity=nonlinearity,
                         hid_init=hid_init,
                         grad_clipping=grad_clipping,
                         precompute_input=precompute_input,
                         mask_input=mask_input,
                         # 1x1 Conv layer for dimensional reduction
                         conv_dim_red=conv_dim_red,
                         conv_dim_red_nonlinearity=conv_dim_red_nonlinearity,
                         # GRU specific params
                         gru_resetgate=gru_resetgate,
                         gru_updategate=gru_updategate,
                         gru_hidden_update=gru_hidden_update,
                         gru_hid_init=gru_hid_init,
                         # LSTM specific params
                         lstm_ingate=lstm_ingate,
                         lstm_forgetgate=lstm_forgetgate,
                         lstm_cell=lstm_cell,
                         lstm_outgate=lstm_outgate,
                         # RNN specific params
                         rnn_W_in_to_hid=rnn_W_in_to_hid,
                         rnn_W_hid_to_hid=rnn_W_hid_to_hid,
                         rnn_b=rnn_b,
                         # Special layers
                         batch_norm=batch_norm,
                         name='reseg')

    # Dynamic cropping
    target_size = get_output(l_in).shape[2:]
    crop = get_output(l_reseg).shape[2:] - target_size
    l_out = CropLayer(l_reseg, crop, centered=False)

    # channel = nclasses
    l_out = lasagne.layers.Conv2DLayer(
        l_out,
        num_filters=nclasses,
        filter_size=(1, 1),
        stride=(1, 1),
        W=out_W_init,
        b=out_b_init,
        nonlinearity=None
    )
    # Go to b01c
    l_out = lasagne.layers.DimshuffleLayer(
        l_out,
        [0, 2, 3, 1],
        name='dimshuffle_before_softmax')

    # Reshape in 2D, last dimension is nclasses, where the softmax is applied
    l_out_shape = get_output(l_out).shape
    l_out = lasagne.layers.ReshapeLayer(
        l_out,
        (T.prod(l_out_shape[0:3]), l_out_shape[3]),
        name='reshape_before_softmax')

    l_out = lasagne.layers.NonlinearityLayer(
        l_out,
        nonlinearity=lasagne.nonlinearities.softmax,
        name="softmax_layer")

    return l_out


def getFunctions(input_var, target_var, class_balance_w_var, l_pred,
                 batch_norm=False, weight_decay=0.,
                 optimizer=lasagne.updates.adadelta,
                 learning_rate=None, momentum=None,
                 rho=None, beta1=None, beta2=None, epsilon=None, ):
    '''Helper function to build the training function

    '''
    input_shape = input_var.shape
    # Compute BN params for prediction
    batch_norm_params = dict()
    if batch_norm:
        batch_norm_params.update(
            dict(batch_norm_update_averages=False))
        batch_norm_params.update(
            dict(batch_norm_use_averages=True))

    # Prediction function:
    # computes the deterministic distribution over the labels, i.e. we
    # disable the stochastic layers such as Dropout
    prediction = lasagne.layers.get_output(l_pred, deterministic=True,
                                           **batch_norm_params)
    f_pred = theano.function(
        [input_var],
        T.argmax(prediction, axis=1).reshape(
            (-1, input_shape[1], input_shape[2])))

    # Compute the loss to be minimized during training
    batch_norm_params = dict()
    if batch_norm:
        batch_norm_params.update(
            dict(batch_norm_update_averages=True))
        batch_norm_params.update(
            dict(batch_norm_use_averages=False))

    prediction = lasagne.layers.get_output(l_pred,
                                           **batch_norm_params)
    loss = lasagne.objectives.categorical_crossentropy(
        prediction, target_var)

    loss *= class_balance_w_var
    loss = loss.reshape((-1, input_shape[1] * input_shape[2]))
    # Compute the cumulative loss (over the pixels) per minibatch
    loss = T.sum(loss, axis=1)
    # Compute the mean loss
    loss = T.mean(loss, axis=0)

    if weight_decay > 0:
        l2_penalty = lasagne.regularization.regularize_network_params(
            l_pred,
            lasagne.regularization.l2,
            tags={'regularizable': True})
        loss += l2_penalty * weight_decay

    params = lasagne.layers.get_all_params(l_pred, trainable=True)

    opt_params = dict()

    if optimizer.__name__ == 'sgd':
        if learning_rate is None:
            raise TypeError("Learning rate can't be 'None' with SGD")
        opt_params = dict(learning_rate=learning_rate)

    elif (optimizer.__name__ == 'momentum' or
          optimizer.__name__ == 'nesterov_momentum'):
        if learning_rate is None:
            raise TypeError("Learning rate can't be 'None' "
                            "with Momentum SGD or Nesterov Momentum")
        opt_params = dict(
            learning_rate=learning_rate,
            momentum=momentum
        )

    elif optimizer.__name__ == 'adagrad':

        if learning_rate is not None:
            opt_params.update(dict(learning_rate=learning_rate))
        if epsilon is not None:
            opt_params.update(dict(epsilon=epsilon))

    elif (optimizer.__name__ == 'rmsprop' or
          optimizer.__name__ == 'adadelta'):

        if learning_rate is not None:
            opt_params.update(dict(learning_rate=learning_rate))
        if rho is not None:
            opt_params.update(dict(rho=rho))
        if epsilon is not None:
            opt_params.update(dict(epsilon=epsilon))

    elif (optimizer.__name__ == 'adam' or
          optimizer.__name__ == 'adamax'):

        if learning_rate is not None:
            opt_params.update(dict(learning_rate=learning_rate))
        if beta1 is not None:
            opt_params.update(dict(beta1=beta1))
        if beta2 is not None:
            opt_params.update(dict(beta2=beta2))
        if epsilon is not None:
            opt_params.update(dict(epsilon=epsilon))

    else:
        raise NotImplementedError('Optimization method not implemented')

    updates = optimizer(loss, params, **opt_params)

    # Training function:
    # computes the training loss (with stochasticity, if any) and
    # updates the weights using the updates dictionary provided by the
    # optimization function
    f_train = theano.function([input_var, target_var, class_balance_w_var],
                              loss, updates=updates)

    return f_pred, f_train


def train(saveto='model.npz',
          tmp_saveto=None,

          # Input Conv layers
          in_nfilters=None,  # None = no input convolution
          in_filters_size=(),
          in_filters_stride=(),
          in_W_init=lasagne.init.GlorotUniform(),
          in_b_init=lasagne.init.Constant(0.),
          in_nonlinearity=lasagne.nonlinearities.rectify,

          # RNNs layers
          dim_proj=[32, 32],
          pwidth=2,
          pheight=2,
          stack_sublayers=(True, True),
          RecurrentNet=lasagne.layers.GRULayer,
          nonlinearity=lasagne.nonlinearities.rectify,
          hid_init=lasagne.init.Constant(0.),
          grad_clipping=0,
          precompute_input=True,
          mask_input=None,

          # 1x1 Conv layer for dimensional reduction
          conv_dim_red=False,
          conv_dim_red_nonlinearity=lasagne.nonlinearities.identity,

          # GRU specific params
          gru_resetgate=lasagne.layers.Gate(W_cell=None),
          gru_updategate=lasagne.layers.Gate(W_cell=None),
          gru_hidden_update=lasagne.layers.Gate(
              W_cell=None,
              nonlinearity=lasagne.nonlinearities.tanh),
          gru_hid_init=lasagne.init.Constant(0.),

          # LSTM specific params
          lstm_ingate=lasagne.layers.Gate(),
          lstm_forgetgate=lasagne.layers.Gate(),
          lstm_cell=lasagne.layers.Gate(
              W_cell=None,
              nonlinearity=lasagne.nonlinearities.tanh),
          lstm_outgate=lasagne.layers.Gate(),

          # RNN specific params
          rnn_W_in_to_hid=lasagne.init.Uniform(),
          rnn_W_hid_to_hid=lasagne.init.Uniform(),
          rnn_b=lasagne.init.Constant(0.),

          # Output upsampling layers
          out_upsampling='grad',
          out_nfilters=None,  # The last number should be the num of classes
          out_filters_size=(1, 1),
          out_filters_stride=None,
          out_W_init=lasagne.init.GlorotUniform(),
          out_b_init=lasagne.init.Constant(0.),
          out_nonlinearity=lasagne.nonlinearities.rectify,

          # Prediction, Softmax
          intermediate_pred=None,
          class_balance=None,

          # Special layers
          batch_norm=False,
          use_dropout=False,
          dropout_rate=0.5,
          use_dropout_x=False,
          dropout_x_rate=0.8,

          # Optimization method
          optimizer=lasagne.updates.adadelta,
          learning_rate=None,
          momentum=None,
          rho=None,
          beta1=None,
          beta2=None,
          epsilon=None,
          weight_decay=0.,  # l2 reg
          weight_noise=0.,

          # Early stopping
          patience=500,  # Num updates with no improvement before early stop
          max_epochs=5000,
          min_epochs=100,

          # Sampling and validation params
          validFreq=1000,
          saveFreq=1000,  # Parameters pickle frequency
          n_save=-1,  # If n_save is a list of indexes, the corresponding
                      # elements of each split are saved. If n_save is an
                      # integer, n_save random elements for each split are
                      # saved. If n_save is -1, all the dataset is saved

          # Batch params
          batch_size=8,
          valid_batch_size=1,
          shuffle=True,

          # Dataset
          dataset='horses',
          color_space='RGB',
          color=True,
          use_depth=None,
          resize_images=True,
          resize_size=-1,

          # Pre-processing
          preprocess_type=None,
          patch_size=(9, 9),
          max_patches=1e5,

          # Data augmentation
          do_random_flip=False,
          do_random_shift=False,
          do_random_invert_color=False,
          shift_pixels=2,
          reload_=False
          ):

    # Set options and history_acc
    # ----------------------------
    start = time.time()  # we use time.time() to know the *real-world* time
    bestparams = {}

    rng = np.random.RandomState(0xbeef)
    saveto = [tmp_saveto, saveto] if tmp_saveto else [saveto]
    if type(pwidth) != list:
        pwidth = [pwidth] * len(dim_proj)
    if type(pheight) != list:
        pheight = [pheight] * len(dim_proj)
    # TODO Intermediate pred should probably have length nlayer - 1,
    # i.e., we don't need to enforce the last one to be True
    # TODO We are not using it for now
    # if intermediate_pred is None:
    #     intermediate_pred = [[False] * (len(dim_proj) - 1)] + [[False, True]]
    # if not unroll(intermediate_pred)[-1]:
    #    raise ValueError('The last value of intermediate_pred should be True')
    if not resize_images and valid_batch_size != 1:
        raise ValueError('When images are not resized valid_batch_size'
                         'should be 1')
    color = color if color else False
    nchannels = 3 if color else 1
    mode = None
    if nanguard:
        mode = NanGuardMode(nan_is_error=True, inf_is_error=True,
                            big_is_error=True)
    options = locals().copy()

    # Repositories hash
    options['recseg_version'] = check_output('git rev-parse HEAD', shell=True)
    options['lasagne_version'] = lasagne.__version__
    options['theano_version'] = theano.__version__

    # options['trng'] = [el[0].get_value() for el in trng.state_updates]
    options['history_acc'] = np.array([])
    options['history_conf_matrix'] = np.array([])
    options['history_iou_index'] = np.array([])

    # Reload
    # ------
    if reload_:
        for s in saveto[::-1]:
            try:
                with open('%s.pkl' % s, 'rb') as f:
                    options_reloaded = pkl.load(f)
                    for k, v in options.iteritems():
                        if k in ['trng', 'history_acc',
                                 'history_conf_matrix',
                                 'history_iou_index']:
                            continue
                        if k not in options_reloaded:
                            print('{} was not present in the options '
                                  'file'.format(k))
                        options_reloaded[k] = v
                    options = options_reloaded
                    print('Option file loaded: {}'.format(s))
                break
            except IOError:
                continue

    saveto = options['saveto']

    # Input Conv layers
    in_nfilters = options['in_nfilters']
    in_filters_size = options['in_filters_size']
    in_filters_stride = options['in_filters_stride']
    in_W_init = options['in_W_init']
    in_b_init = options['in_b_init']
    in_nonlinearity = options['in_nonlinearity']

    # RNNs layers
    dim_proj = options['dim_proj']
    pwidth = options['pwidth']
    pheight = options['pheight']
    stack_sublayers = options['stack_sublayers']
    RecurrentNet = options['RecurrentNet']
    nonlinearity = options['nonlinearity']
    hid_init = options['hid_init']
    grad_clipping = options['grad_clipping']
    precompute_input = options['precompute_input']
    mask_input = options['mask_input']

    # 1x1 Conv layer for dimensional reduction
    conv_dim_red = options['conv_dim_red']
    conv_dim_red_nonlinearity = options['conv_dim_red_nonlinearity']

    # GRU specific params
    gru_resetgate = options['gru_resetgate']
    gru_updategate = options['gru_updategate']
    gru_hidden_update = options['gru_hidden_update']
    gru_hid_init = options['gru_hid_init']

    # LSTM specific params
    lstm_ingate = options['lstm_ingate']
    lstm_forgetgate = options['lstm_forgetgate']
    lstm_cell = options['lstm_cell']
    lstm_outgate = options['lstm_outgate']

    # RNN specific params
    rnn_W_in_to_hid = options['rnn_W_in_to_hid']
    rnn_W_hid_to_hid = options['rnn_W_hid_to_hid']
    rnn_b = options['rnn_b']

    # Output upsampling layers
    out_upsampling = options['out_upsampling']
    out_nfilters = options['out_nfilters']
    out_filters_size = options['out_filters_size']
    out_filters_stride = options['out_filters_stride']
    out_W_init = options['out_W_init']
    out_b_init = options['out_b_init']
    out_nonlinearity = options['out_nonlinearity']

    # Prediction, Softmax
    intermediate_pred = options['intermediate_pred']
    class_balance = options['class_balance']

    # Special layers
    batch_norm = options['batch_norm']
    use_dropout = options['use_dropout']
    dropout_rate = options['dropout_rate']
    use_dropout_x = options['use_dropout_x']
    dropout_x_rate = options['dropout_x_rate']

    # Optimization method
    optimizer = options['optimizer']
    learning_rate = options['learning_rate']
    momentum = options['momentum']
    rho = options['rho']
    beta1 = options['beta1']
    beta2 = options['beta2']
    epsilon = options['epsilon']
    weight_decay = options['weight_decay']
    weight_noise = options['weight_noise']

    # Batch params
    batch_size = options['batch_size']
    valid_batch_size = options['valid_batch_size']
    shuffle = options['shuffle']

    # Dataset
    dataset = options['dataset']
    color_space = options['color_space']
    color = options['color']
    use_depth = options['use_depth']
    resize_images = options['resize_images']
    resize_size = options['resize_size']

    # Pre-processing
    preprocess_type = options['preprocess_type']
    patch_size = options['patch_size']
    max_patches = options['max_patches']

    # Data augmentation
    do_random_flip = options['do_random_flip']
    do_random_shift = options['do_random_shift']
    do_random_invert_color = options['do_random_invert_color']
    shift_pixels = options['shift_pixels']

    # Save state from options
    rng = options['rng']
    # trng = options['trng'] --> to be reloaded after building the model
    history_acc = options['history_acc'].tolist()
    history_conf_matrix = options['history_conf_matrix'].tolist()
    history_iou_index = options['history_iou_index'].tolist()
    print_params(options)

    n_layers = len(dim_proj)

    assert class_balance in [None, 'median_freq_cost', 'natural_freq_cost',
                             'priors_correction'], (
        'The balance class method is not implemented')
    assert (preprocess_type in [None, 'f-whiten', 'conv-zca', 'sub-lcn',
                                'subdiv-lcn', 'gcn', 'local_mean_sub']), (
            "The preprocessing method choosen is not implemented")

    # Load data
    # ---------
    print("Loading data ...")
    load_data, properties = get_dataset(dataset)
    train, valid, test, mean, std, filenames, fullmasks = load_data(
        resize_images=resize_images,
        resize_size=resize_size,
        color=color,
        color_space=color_space,
        rng=rng,
        use_depth=use_depth,
        with_filenames=True,
        with_fullmasks=True)
    has_void_class = properties()['has_void_class']

    if not color:
        if mean.ndim == 3:
            mean = np.expand_dims(mean, axis=3)
        if std.ndim == 3:
            std = np.expand_dims(std, axis=3)

    # Preprocess each image separately usually with LCN in order not to lose
    # time at each epoch
    train, valid, test = preprocess_dataset(train, valid, test,
                                            preprocess_type,
                                            patch_size, max_patches)

    # Compute the indexes of the images to be saved
    if isinstance(n_save, collections.Iterable):
        samples_ids = np.array(n_save)
    elif n_save != -1:
        samples_ids = [
            random.sample(range(len(s)), min(len(s), n_save)) for s in
            [train[0], valid[0], test[0]]]
    else:
        samples_ids = [range(len(s)) for s in [train[0], valid[0], test[0]]]
    options['samples_ids'] = samples_ids

    # Retrieve basic size informations and split train variables
    x_train, y_train = train
    if len(x_train) == 0:
        raise RuntimeError("Dataset not found")
    filenames_train, filenames_valid, filenames_test = filenames
    cheight, cwidth, cchannels = x_train[0].shape
    nclasses = max([np.max(el) for el in y_train]) + 1
    print '# of classes:', nclasses

    # Remove the segmentation samples dir to make sure we don't mix samples
    # from different experiments
    seg_path = os.path.join('segmentations', dataset,
                            saveto[0].split('/')[-1][:-4])
    try:
        rmtree(seg_path)
    except OSError:
        pass

    # Class balancing
    # ---------------
    # TODO: check if it works...
    w_freq = 1
    if class_balance in ['median_freq_cost', 'rare_freq_cost']:
        u_train, c_train = np.unique(y_train, return_counts=True)
        priors = c_train.astype(theano.config.floatX) / train[1].size

        # the denominator is computed by summing the total number
        # of pixels of the images where the class is present
        # so it should be even more balanced
        px_count = np.zeros(u_train.shape)
        for tt in y_train:
            u_tt = np.unique(tt)
            px_t = tt.size
            for uu in u_tt:
                px_count[uu] += px_t
        priors = c_train.astype(theano.config.floatX) / px_count

        if class_balance == 'median_freq_cost':
            w_freq = np.median(priors) / priors
        elif class_balance == 'rare_freq_cost':
            w_freq = 1 / (nclasses * priors)

        print "Class balance weights", w_freq

        assert len(priors) == nclasses, ("Number of computed priors are "
                                         "different from number of classes")

    if validFreq == -1:
        validFreq = len(x_train)/batch_size
    if saveFreq == -1:
        saveFreq = len(x_train)/batch_size

    # Model compilation
    # -----------------
    print("Building model ...")

    input_shape = (None, cheight, cwidth, cchannels)
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    class_balance_w_var = T.vector('class_balance_w_var')

    # Set the RandomStream to assure repeatability
    lasagne.random.set_rng(rng)

    # Tag test values
    if debug:
        print "DEBUG MODE: loading tag.test_value ..."
        load_data, properties = get_dataset(dataset)
        train, _, _, _, _ = load_data(
            resize_images=resize_images, resize_size=resize_size,
            color=color, color_space=color_space, rng=rng)

        x_tag = (train[0][0:batch_size]).astype(floatX)
        y_tag = (train[1][0:batch_size]).astype(intX)

        # TODO Move preprocessing in a separate function
        if x_tag.ndim == 1:
            x_tag = x_tag[0]
            y_tag = y_tag[0]
        if x_tag.ndim == 3:
            x_tag = np.expand_dims(x_tag, 0)
            y_tag = np.expand_dims(y_tag, 0)

        input_var.tag.test_value = x_tag
        target_var.tag.test_value = y_tag.flatten()
        class_balance_w_var.tag.test_value = np.ones(
            np.prod(x_tag.shape[:3])).astype(floatX)
        theano.config.compute_test_value = 'warn'

    # Build the model
    l_out = buildReSeg(input_shape, input_var,
                       n_layers, pheight, pwidth,
                       dim_proj, nclasses, stack_sublayers,
                       # upsampling
                       out_upsampling,
                       out_nfilters,
                       out_filters_size,
                       out_filters_stride,
                       out_W_init=out_W_init,
                       out_b_init=out_b_init,
                       out_nonlinearity=out_nonlinearity,
                       # input ConvLayers
                       in_nfilters=in_nfilters,
                       in_filters_size=in_filters_size,
                       in_filters_stride=in_filters_stride,
                       in_W_init=in_W_init,
                       in_b_init=in_b_init,
                       in_nonlinearity=in_nonlinearity,
                       # common recurrent layer params
                       RecurrentNet=RecurrentNet,
                       nonlinearity=nonlinearity,
                       hid_init=hid_init,
                       grad_clipping=grad_clipping,
                       precompute_input=precompute_input,
                       mask_input=mask_input,
                       # 1x1 Conv layer for dimensional reduction
                       conv_dim_red=conv_dim_red,
                       conv_dim_red_nonlinearity=conv_dim_red_nonlinearity,
                       # GRU specific params
                       gru_resetgate=gru_resetgate,
                       gru_updategate=gru_updategate,
                       gru_hidden_update=gru_hidden_update,
                       gru_hid_init=gru_hid_init,
                       # LSTM specific params
                       lstm_ingate=lstm_ingate,
                       lstm_forgetgate=lstm_forgetgate,
                       lstm_cell=lstm_cell,
                       lstm_outgate=lstm_outgate,
                       # RNN specific params
                       rnn_W_in_to_hid=rnn_W_in_to_hid,
                       rnn_W_hid_to_hid=rnn_W_hid_to_hid,
                       rnn_b=rnn_b,
                       # special layers
                       batch_norm=batch_norm)

    f_pred, f_train = getFunctions(input_var, target_var, class_balance_w_var,
                                   l_out, weight_decay, optimizer=optimizer,
                                   learning_rate=learning_rate,
                                   momentum=momentum, rho=rho, beta1=beta1,
                                   beta2=beta2, epsilon=epsilon)

    # Reload the list of the value parameters
    # TODO Check if the saved params are CudaNDArrays or not, so that we
    # don't need a GPU to reload the model (I'll do it when you are
    # done)
    if reload_:
        for s in saveto[::-1]:
            try:
                with np.load('%s' % s) as f:
                    vparams = [f['arr_%d' % i] for i in range(len(f.files))]
                    lastparams, bestparams = vparams
                    # for i, v in enumerate(options['trng']):
                    #     trng.state_updates[i][0].set_value(v)
                    print('Model file loaded: {}'.format(s))
                lasagne.layers.set_all_param_values(l_out, bestparams)

                break
            except IOError:
                continue

    # Main loop
    # ---------
    print("Starting training...")
    uidx = 0
    patience_counter = 0
    estop = False
    save = False

    epochs_wid = VariableText(
        'Epoch %(epoch)d/' + str(max_epochs) + ' Up %(up)d',
        {'epoch': 0, 'up': 0})
    metrics_wid = VariableText(
        'Cost %(cost)f, DD %(DD)f, UD %(UD)f %(shape)s',
        {'cost': None,
         'DD': None,
         'UD': None,
         'shape': None})
    widgets = [
        '', epochs_wid,
        ' ', metrics_wid]
    pbar = ProgressBar(widgets=widgets, maxval=len(x_train),
                       redirect_stdout=True).start()

    # Epochs loop
    for eidx in range(max_epochs):
        nsamples = 0
        epoch_cost = 0
        start_time = time.time()

        # Minibatches loop
        for i, minibatch in enumerate(iterate_minibatches(x_train,
                                                          y_train,
                                                          batch_size,
                                                          rng=rng,
                                                          shuffle=shuffle)):
            inputs, targets, _ = minibatch
            st = time.time()
            nsamples += len(inputs)
            uidx += 1

            # otherwise the normalization has been done before the preprocess
            # if preprocess_type is None:
            #     inputs = inputs.astype(floatX)

            targets = targets.astype(intX)
            targets_flat = targets.flatten()

            dd = time.time() - st
            st = time.time()

            # Class balance
            class_balance_w = np.ones(np.prod(inputs.shape[:3])).astype(floatX)
            if class_balance in ['median_freq_cost', 'rare_freq_cost']:
                class_balance_w = w_freq[targets_flat].astype(floatX)

            # Compute cost
            cost = f_train(inputs, targets_flat, class_balance_w)
            ud = time.time() - st

            if np.isnan(cost):
                raise RuntimeError('NaN detected')
            if np.isinf(cost):
                raise RuntimeError('Inf detected')

            # if np.mod(uidx, dispFreq) == 0:
            #     print('Epoch {}, Up {}, Cost {:.3f}, DD {:.3f}, UD ' +
            #           '{:.5f} {}').format(eidx, uidx, float(cost), dd, ud,
            #                               input_shape)
            epochs_wid.update_mapping({'epoch': eidx, 'up': uidx})
            metrics_wid.update_mapping(
                {'cost': float(cost),
                 'DD': dd,
                 'UD': ud,
                 'shape': input_shape})
            pbar.update(min(i*batch_size + 1, len(x_train)))

            def validate_model():
                (train_global_acc,
                 train_conf_matrix,
                 train_mean_class_acc,
                 train_iou_index,
                 train_mean_iou_index) = validate(f_pred,
                                                  train,
                                                  valid_batch_size,
                                                  has_void_class,
                                                  preprocess_type,
                                                  nclasses,
                                                  samples_ids=samples_ids[0],
                                                  filenames=filenames_train,
                                                  folder_dataset='train',
                                                  dataset=dataset,
                                                  saveto=saveto[0])
                (valid_global_acc,
                 valid_conf_matrix,
                 valid_mean_class_acc,
                 valid_iou_index,
                 valid_mean_iou_index) = validate(f_pred,
                                                  valid,
                                                  valid_batch_size,
                                                  has_void_class,
                                                  preprocess_type,
                                                  nclasses,
                                                  samples_ids=samples_ids[1],
                                                  filenames=filenames_valid,
                                                  folder_dataset='valid',
                                                  dataset=dataset,
                                                  saveto=saveto[0])
                (test_global_acc,
                 test_conf_matrix,
                 test_mean_class_acc,
                 test_iou_index,
                 test_mean_iou_index) = validate(f_pred,
                                                 test,
                                                 valid_batch_size,
                                                 has_void_class,
                                                 preprocess_type,
                                                 nclasses,
                                                 samples_ids=samples_ids[2],
                                                 filenames=filenames_test,
                                                 folder_dataset='test',
                                                 dataset=dataset,
                                                 saveto=saveto[0])
                print("")
                print("Global Accuracies:")
                print('Train {:.5f} Valid {:.5f} Test {:.5f}'.format(
                    train_global_acc, valid_global_acc, test_global_acc))

                print("Mean Class Accuracy:")
                print('Train {:.5f} Valid {:.5f} Test {:.5f}'.format(
                    train_mean_class_acc, valid_mean_class_acc,
                    test_mean_class_acc))

                print('Train {:.5f} Valid {:.5f} Test {:.5f}'.format(
                    train_mean_iou_index, valid_mean_iou_index,
                    test_mean_iou_index))
                print("")

                history_acc.append([train_global_acc,
                                    train_mean_class_acc,
                                    train_mean_iou_index,
                                    valid_global_acc,
                                    valid_mean_class_acc,
                                    valid_mean_iou_index,
                                    test_global_acc,
                                    test_mean_class_acc,
                                    test_mean_iou_index])

                history_conf_matrix.append([train_conf_matrix,
                                           valid_conf_matrix,
                                           test_conf_matrix])

                history_iou_index.append([train_iou_index,
                                         valid_iou_index,
                                         test_iou_index])

                options['history_acc'] = np.array(history_acc)
                options['history_conf_matrix'] = np.array(history_conf_matrix)
                options['history_iou_index'] = np.array(history_iou_index)

                return valid_global_acc, test_global_acc

            # Check predictions' accuracy
            if np.mod(uidx, validFreq) == 0:
                valid_global_acc, test_global_acc = validate_model()

                # Did we improve *validation* accuracy?
                if (len(valid) > 0 and
                    (len(history_acc) == 0 or valid_global_acc >= np.array(
                        history_acc)[:, 3].max())):

                    # TODO check if CUDA variables!
                    bestparams = lasagne.layers.get_all_param_values(l_out)
                    patience_counter = 0
                    save = True  # Save model params
                else:
                    # if validation set is empty check test set to save params
                    if len(history_acc) == 0 or test_global_acc >= np.array(
                            history_acc)[:, 6].max():
                        # TODO check if CUDA variables!
                        bestparams = lasagne.layers.get_all_param_values(
                            l_out)
                        patience_counter = 0
                        # Save model params
                        save = True

                # Early stop if patience is over
                if (eidx > min_epochs):
                    patience_counter += 1
                    if patience_counter == patience / validFreq:
                        print 'Early Stop!'
                        estop = True

            # Save model parameters
            if save or np.mod(uidx, saveFreq) == 0:
                save_time = time.time()
                lastparams = lasagne.layers.get_all_param_values(l_out)
                vparams = [lastparams, bestparams]
                # Retry if filesystem is busy
                save_with_retry(saveto[0], vparams)
                save = False
                pkl.dump(options,
                         open('%s.pkl' % saveto[0], 'wb'))
                print 'Saved parameters and options in {} in {:.3f}s'.format(
                    saveto[0], time.time() - save_time)

            epoch_cost += cost

            # exit minibatches loop
            if estop:
                break

        # exit epochs loop
        if estop:
            break

        print("Epoch {} of {} took {:.3f}s with overall cost {:.3f}".format(
            eidx + 1, max_epochs, time.time() - start_time, epoch_cost))

    pbar.finish()
    validate_model()
    max_valid_idx = np.argmax(np.array(history_acc)[:, 3])
    best = history_acc[max_valid_idx]
    print("Global Accuracies :")
    print('Best: Train {:.5f} Valid {:.5f} Test {:.5f}'.format(
        best[0], best[3], best[6]))
    print("Test Mean Class Accuracy: {}".format(best[7]))
    print("Test Mean Intersection Over Union: {}".format(best[8]))

    if len(saveto) != 1:
        print("Moving temporary model files to {}".format(saveto[1]))
        dirname = os.path.dirname(saveto[1])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        move(saveto[0], saveto[1])
        move(saveto[0] + '.pkl', saveto[1] + '.pkl')

    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("Total time elapsed: %d:%02d:%02d" % (h, m, s))
    return best


def show_seg(dataset_name, n_exp, dataset_set, mode='sequential', id=-1):
    """

    :param model_filename: model_recseg_namedataset1.npz
    :param dataset_set: 'train', 'valid','test'
    :param mode: 'random', 'sequential', 'filename', 'id'
    :param id: 'filename' or 'index'
    :return:
    """

    load_from_file = True
    # load options
    model_filename = 'model_recseg_' + dataset_name + n_exp + ".npz"
    options = pkl.load(open(
            os.path.expanduser(
                    os.path.join(dataset_name + "_models",
                                 model_filename + '.pkl')), 'rb'))

    # now go in the default directory to retrieve all the prediction masks
    name = options['dataset']

    print("Loading data ...")
    load_data, properties = get_dataset(options['dataset'])
    train, valid, test, mean, std, filenames, fullmasks = load_data(
        resize_images=options['resize_images'],
        resize_size=options['resize_size'],
        color=options['color'],
        color_space=options['color_space'],
        rng=options['rng'],
        use_depth=options['use_depth'],
        with_filenames=True,
        with_fullmasks=True)

    if load_from_file:
        seg_path = os.path.abspath(
                os.path.join('segmentations', name,
                             model_filename.split('/')[-1][:-4]))

    if dataset_set == 'train':
        id_f = 0
        images = train[0]
        gt = train[1]
    elif dataset_set == 'valid':
        id_f = 1
        images = valid[0]
        gt = valid[1]
    else:
        id_f = 2
        images = test[0]
        gt = test[1]

    # TODO: use buildReSeg()
    out_layer = [id_f, images, gt, seg_path]  # DELETEME!

    # load params
    with np.load('model.npz') as f:
        bestparams_val = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(out_layer, bestparams_val)

    # compute prediction on the dataset or on the image that we specified

if __name__ == '__main__':

    if len(sys.argv) >= 3:
        dataset_name = sys.argv[1]
        n_exp = sys.argv[2]

    else:
        print "Usage: dataset_name n_exp, e.g. python reseg.py camvid 1"
        sys.exit()

    if len(sys.argv) > 3:
        if sys.argv[3] in ['train', 'valid', 'test']:
            dataset_set = sys.argv[3]
        else:
            print "Usage: choose one between 'train', 'valid', 'test'"
            sys.exit()
    else:
        dataset_set = 'test'

    if len(sys.argv) > 4:
        if sys.argv[4] in ['random', 'sequential', 'filename', 'id']:
            mode = sys.argv[4]
            if mode in ['filename', 'id']:
                if len(sys.argv) < 6:
                    print "Insert a correct filename or id!"
                    sys.exit()
                else:
                    id = sys.argv[5]
            else:
                id = -1
        else:
            print "Usage: mode can be 'random', 'sequential', 'filename', 'id'"
            sys.exit()
    else:
        mode = 'sequential'

    show_seg(dataset_name, n_exp, dataset_set)
