# Standard library imports
import cPickle as pkl
import pickle
import collections
import matplotlib.pyplot as plt
import os
from shutil import move
import sys
import time

# Related third party imports
import lasagne
import numpy as np
from skimage.data import load
from skimage.color import label2rgb
import theano
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode
# from vgg16 import build_model as buildVgg16

# Local application/library specific imports
from config_datasets import color_list_datasets
from get_info_model import print_params
from layers import ReSegLayer
from subprocess import check_output
from utils import iterate_minibatches, validate, save_with_retry, unroll
from vgg16 import build_model as buildVgg16

# Datasets import
# TODO these should go into preprocess/helper dataset
import camvid
import daimler
import fashionista
import oxford_flowers
import kitti_road

floatX = theano.config.floatX
intX = 'uint8'

debug = False
nanguard = False

datasets = {'camvid': (camvid.load_data, camvid.properties),
            'daimler': (daimler.load_data, daimler.properties),
            'fashionista': (fashionista.load_data, fashionista.properties),
            'flowers': (oxford_flowers.load_data, oxford_flowers.properties),
            'kitti_road': (kitti_road.load_data, kitti_road.properties)}


def get_dataset(name):
    return (datasets[name][0], datasets[name][1])


def buildReSeg(input_shape, input_var,
               n_layers, pheight, pwidth, dim_proj, nclasses,
               stack_sublayers, out_upsampling, out_nfilters, out_filters_size,
               out_filters_stride):
    '''Helper function to build a ReSeg network'''

    print('Input shape: ' + str(input_shape))
    l_in = lasagne.layers.InputLayer(shape=input_shape,
                                     input_var=input_var,
                                     name="input_layer")

    # Pretrained vgg16
    use_pretrained_vgg = True
    if use_pretrained_vgg:
        # Convert to batchsize, ch, rows, cols
        l_in = lasagne.layers.DimshuffleLayer(l_in, (0, 3, 1, 2))
        l_vgg16 = buildVgg16(l_in, False, False)['conv5_3']
        # Reload weights
        vgg16_w = pickle.load(open('vgg16.pkl'))
        lasagne.layers.set_all_param_values(l_vgg16,
                                            vgg16_w['param values'][:-6])
        # Back to batchsize, rows, cols, ch
        l_in = lasagne.layers.DimshuffleLayer(l_vgg16, (0, 2, 3, 1))

    l_reseg = ReSegLayer(l_in, n_layers, pheight, pwidth, dim_proj,
                         nclasses, stack_sublayers, out_upsampling,
                         out_nfilters, out_filters_size,
                         out_filters_stride, 'reseg')
    # Reshape in 2D, last dimension is nclasses, where the softmax is applied
    l_out = lasagne.layers.ReshapeLayer(
        l_reseg,
        (T.prod(l_reseg.output_shape[0:3]), l_reseg.output_shape[3]),
        name='reshape_before_softmax')

    l_pred = lasagne.layers.NonlinearityLayer(
        l_out,
        nonlinearity=lasagne.nonlinearities.softmax,
        name="softmax_layer")

    # Compile the function that gives back the mask prediction
    # with deterministic=True we exclude stochastic layers such as dropout
    prediction = lasagne.layers.get_output(l_pred, deterministic=True)
    f_pred = theano.function(
        [input_var],
        T.argmax(prediction, axis=1).reshape(input_shape[:3]))

    return l_pred, f_pred


def buildTrain(input_var, target_var, weights_loss, l_pred, weight_decay=0.):
    '''Helper function to build the training function

    '''
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(l_pred)
    loss = lasagne.objectives.categorical_crossentropy(
        prediction, target_var)

    if weight_decay > 0:
        l2_penalty = lasagne.regularization.regularize_network_params(
            l_pred,
            lasagne.regularization.l2,
            tags={'regularizable': True})
        loss += l2_penalty * weight_decay

    loss = weights_loss * loss.mean()
    params = lasagne.layers.get_all_params(l_pred, trainable=True)
    # Stochastic Gradient Descent (SGD) with Nesterov momentum
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=LEARNING_RATE, momentum=0.9)
    updates = lasagne.updates.adadelta(loss, params)
    # Compile the function that performs a training step on a mini-batch
    # (by using the updates dictionary) and returns the corresponding training
    # loss:
    f_train = theano.function([input_var, target_var, weights_loss], loss,
                              updates=updates)

    return f_train


def train(saveto='model.npz',
          tmp_saveto=None,

          # Input Conv layers
          in_nfilters=None,  # None = no input convolution
          in_filters_size=[],
          in_filters_stride=[],
          in_init='glorot',
          in_activ='tanh',

          # RNNs layers
          encoder='gru',
          dim_proj=[32, 32],
          pwidth=2,
          pheight=2,
          rnn_init='norm_weight',
          rnn_activation='tanh',
          stack_sublayers=(True, True),
          clip_grad_threshold=0.,

          # Output upsampling layers
          out_upsampling='grad',
          out_nfilters=None,  # The last number should be the num of classes
          out_filters_size=(1, 1),
          out_filters_stride=None,
          out_init='glorot',
          out_activ='identity',

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
          optimizer='rmsprop',
          lrate=0.01,
          weight_decay=0.,  # l2 reg
          weight_noise=0.,

          # Early stopping
          patience=500,  # Num updates with no improvement before early stop
          max_epochs=5000,
          min_epochs=100,

          # Save, display params
          dispFreq=100,
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
          reload_=False,
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
    if intermediate_pred is None:
        intermediate_pred = [[False] * (len(dim_proj) - 1)] + [[False, True]]
    if not unroll(intermediate_pred)[-1]:
        raise ValueError('The last value of intermediate_pred should be True')
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
    in_init = options['in_init'] if 'in_init' in options else 'glorot'
    in_activ = options['in_activ'] if 'in_activ' in options else 'identity'

    # RNNs layers
    encoder = options['encoder']
    dim_proj = options['dim_proj']
    pwidth = options['pwidth']
    pheight = options['pheight']
    rnn_init = options['rnn_init']
    rnn_activation = options['rnn_activation']
    stack_sublayers = options['stack_sublayers']
    clip_grad_threshold = options['clip_grad_threshold']

    # Output upsampling layers
    out_upsampling = options['out_upsampling']
    out_nfilters = options['out_nfilters']
    out_filters_size = options['out_filters_size']
    out_filters_stride = options['out_filters_stride']
    out_init = options['out_init']
    out_activ = options['out_activ']

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
    lrate = options['lrate']
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

    assert class_balance in [None, 'median_freq_cost',
                             'natural_freq_cost',
                             'priors_correction'], ('The balance class '
                                                    'method is not '
                                                    'implemented')
    assert (preprocess_type in
            [None, 'f-whiten',
             'conv-zca',
             'sub-lcn',
             'subdiv-lcn',
             'gcn',
             'local_mean_sub']), ("The preprocessing method choosen is not "
                                  "implemented")

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
        with_filenames=True,
        with_fullmasks=True)

    if not color:
        if mean.ndim == 3:
            mean = np.expand_dims(mean, axis=3)
        if std.ndim == 3:
            std = np.expand_dims(std, axis=3)

    # Compute the indexes of the images to be saved
    if isinstance(n_save, collections.Iterable):
        n_save = np.array(n_save)
    elif isinstance(n_save, int):
        n_save = min(n_save, len(train[0]), len(valid[0]), len(test[0]))
        n_save = np.arange(n_save)
        rng.shuffle(n_save)
    options['n_save'] = n_save

    # Retrieve basic size informations and split train variables
    x_train, y_train = train
    filenames_train, filenames_valid, filenames_test = filenames
    cheight, cwidth, cchannels = x_train[0].shape
    nclasses = max([np.max(el) for el in y_train]) + 1

    # TODO Do we still need it?
    if out_nfilters[-1] != nclasses:
        raise RuntimeError('The last element of out_nfilters should be'
                           '%d' % nclasses)
    print '# of classes:', nclasses

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
        w_freq_shared = theano.shared(w_freq)

    if validFreq == -1:
        validFreq = len(x_train)
    if saveFreq == -1:
        saveFreq = len(x_train)

    # Model compilation
    # -----------------
    print("Building model ...")

    input_shape = (batch_size, cheight, cwidth, cchannels)
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    weights_loss = T.scalar('weights_loss')

    # Set the RandomStream to assure repeatability
    lasagne.random.set_rng(rng)

    # Tag test values
    if debug:
        print "DEBUG MODE: loading tag.test_value ..."
        load_data, properties = get_dataset(dataset)
        train, _, _, _, _ = load_data(
            resize_images=resize_images, resize_size=resize_size,
            color=color, color_space=color_space, rng=rng)
        x_tag = (train[0][0:batch_size] / 255.).astype(floatX)
        y_tag = (train[1][0:batch_size]).astype(intX)

        # Move preprocessing in a separate function
        if x_tag.ndim == 1:
            x_tag = x_tag[0]
            y_tag = y_tag[0]
        if x_tag.ndim == 3:
            x_tag = np.expand_dims(x_tag, 0)
            y_tag = np.expand_dims(y_tag, 0)

        # TODO Do we need it? Reason about this
        # crop the image if it is not a multiple of the patch size
        if 'linear' in out_upsampling:
            dh = x_tag.shape[1] % np.prod(pheight)
            dw = x_tag.shape[2] % np.prod(pwidth)
            x_tag = x_tag[:, dh/2:(-dh+dh/2 if -dh+dh/2 else None),
                          dw/2:(-dw+dw/2 if -dw/dw/2 else None), ...]
            y_tag = y_tag[:, dh/2:(-dh+dh/2 if -dh+dh/2 else None),
                          dw/2:(-dw+dw/2 if -dw/dw/2 else None), ...]

        input_var.tag.test_value = x_tag
        target_var.tag.test_value = y_tag.flatten()
        theano.config.compute_test_value = 'warn'

    # Build the model
    out_layer, f_pred = buildReSeg(input_shape, input_var,
                                   n_layers, pheight, pwidth,
                                   dim_proj, nclasses,
                                   stack_sublayers, out_upsampling,
                                   out_nfilters, out_filters_size,
                                   out_filters_stride)
    f_train = buildTrain(input_var, target_var, weights_loss, out_layer,
                         weight_decay)

    # Reload the list of the value parameters
    # TODO Check if the saved params are CudaNDArrays or not, so that we
    # don't need a GPU to reload the model (I'll do it when you are
    # done)
    if reload_:
        for s in saveto[::-1]:
            try:
                with np.load('%s' % s) as f:
                    vparams = [f['arr_%d' % i] for i in range(len(f.files))]
                    bestparams, bestparams_val = vparams
                    # for i, v in enumerate(options['trng']):
                    #     trng.state_updates[i][0].set_value(v)
                    print('Model file loaded: {}'.format(s))
                lasagne.layers.set_all_param_values(out_layer, bestparams_val)

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

    # Epochs loop
    for eidx in range(max_epochs):
        nsamples = 0
        train_cost = 0
        start_time = time.time()

        # Minibatches loop
        for minibatch in iterate_minibatches(x_train,
                                             y_train,
                                             batch_size,
                                             rng=rng,
                                             shuffle=shuffle):
            inputs, targets, _ = minibatch
            st = time.time()
            nsamples += len(inputs)
            uidx += 1

            inputs = (inputs / 255.).astype(floatX)
            targets = targets.astype(intX)
            # TODO Don't we need shape information?
            targets_flat = targets.flatten()

            # TODO Move to preprocessing fn
            # crop if not a multiple of the patch size
            if 'linear' in out_upsampling:
                dh = inputs.shape[1] % np.prod(options['pheight'])
                dw = inputs.shape[2] % np.prod(options['pwidth'])
                x = inputs[:, dh/2:(-dh+dh/2 if -dh+dh/2 else None),
                           dw/2:(-dw+dw/2 if -dw/dw/2 else None), ...]
                y = targets[:, dh/2:(-dh+dh/2 if -dh+dh/2 else None),
                            dw/2:(-dw+dw/2 if -dw/dw/2 else None), ...]

            print 'Image size: {}'.format(x.shape)

            # TODO: preprocess function
            # whiten, LCN, GCN, Local Mean Subtract, or normalize +
            # stochastically augment data
            # x, y = preprocess(x,
            #                   options['color'], mean, std,
            #                   options['preprocess_type'],
            #                   y, rng, do_random_flip,
            #                   do_random_shift, do_random_invert_color,
            #                   options['reorder'], options['shift_pixels'],
            #                   patch_size, max_patches)

            dd = time.time() - st
            st = time.time()

            # Class balance
            w = 1
            if class_balance in ['median_freq_cost', 'rare_freq_cost']:
                w = np.sum(w_freq[targets_flat]).astype(floatX)

            # Compute cost
            cost = f_train(inputs, targets_flat, w)
            ud = time.time() - st

            if np.isnan(cost):
                raise RuntimeError('NaN detected')
            if np.isinf(cost):
                raise RuntimeError('Inf detected')

            if np.mod(uidx, dispFreq) == 0:
                print('Epoch {}, Update {}, Cost {}, DD {}, UD {}').format(
                        eidx, uidx, round(cost, 5), round(dd), round(ud))

            # Check predictions' accuracy
            if eidx == max_epochs - 1 or np.mod(uidx, validFreq) == 0:

                # NOTE : No need to suppress any stochastic layer such as
                # Dropout, since f_pred exclude any non-deterministic layer
                (train_global_acc,
                 train_conf_matrix,
                 train_conf_matrix_norm,
                 train_mean_class_acc,
                 train_iou_index,
                 train_mean_iou_index) = validate(f_pred,
                                                  train,
                                                  valid_batch_size,
                                                  nclasses,
                                                  rng=rng,
                                                  n_save=n_save,
                                                  filenames=filenames_train,
                                                  folder_dataset='train',
                                                  dataset=dataset,
                                                  saveto=saveto[0])
                (valid_global_acc,
                 valid_conf_matrix,
                 valid_conf_matrix_norm,
                 valid_mean_class_acc,
                 valid_iou_index,
                 valid_mean_iou_index) = validate(f_pred,
                                                  valid,
                                                  valid_batch_size,
                                                  nclasses,
                                                  rng=rng,
                                                  n_save=n_save,
                                                  filenames=filenames_valid,
                                                  folder_dataset='valid',
                                                  dataset=dataset,
                                                  saveto=saveto[0])
                (test_global_acc,
                 test_conf_matrix,
                 test_conf_matrix_norm,
                 test_mean_class_acc,
                 test_iou_index,
                 test_mean_iou_index) = validate(f_pred,
                                                 test,
                                                 valid_batch_size,
                                                 nclasses,
                                                 rng=rng,
                                                 n_save=n_save,
                                                 filenames=filenames_test,
                                                 folder_dataset='test',
                                                 dataset=dataset,
                                                 saveto=saveto[0])

                print("")
                print("Global Accuracies :")
                print('Train ', train_global_acc,
                      'Valid ', valid_global_acc,
                      'Test ', test_global_acc)
                print("")
                print("Class Accuracies :")
                print('Train ', train_mean_class_acc,
                      'Valid ', valid_mean_class_acc,
                      'Test ', test_mean_class_acc)
                print("")
                print("Mean Intersection Over Union :")
                print('Train ', train_mean_iou_index,
                      'Valid ', valid_mean_iou_index,
                      'Test ', test_mean_iou_index)
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

                history_conf_matrix.append([train_conf_matrix_norm,
                                           valid_conf_matrix_norm,
                                           test_conf_matrix_norm,
                                           train_conf_matrix,
                                           valid_conf_matrix,
                                           test_conf_matrix])

                history_iou_index.append([train_iou_index,
                                         valid_iou_index,
                                         test_iou_index])

                options['history_acc'] = np.array(history_acc)
                options['history_conf_matrix'] = np.array(history_conf_matrix)
                options['history_iou_index'] = np.array(history_iou_index)

                # Did we improve *validation* accuracy?
                if len(history_acc) == 0 or valid_global_acc >= np.array(
                        history_acc)[:, 3].max():

                    # TODO check if CUDA variables!
                    bestparams = lasagne.layers.get_all_param_values(out_layer)
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
                print 'Saving the parameters of the model...',
                lastparams = lasagne.layers.get_all_param_values(out_layer)
                vparams = [lastparams, bestparams]
                # Retry if filesystem is busy
                save_with_retry(saveto[0], vparams)
                # if normalization:
                #     numpy.savez('%s.norm.npz' % saveto,
                #                 values=[np.get_value() for np in
                #                         nparams[0] + nparams[1]])
                save = False
                print 'Done in {:.3f}s'.format(time.time() - save_time)

                print "Saving the options to {}".format(saveto[0])
                pkl.dump(options,
                         open('%s.pkl' % saveto[0], 'wb'))

            train_cost += cost

            # exit minibatchs loop
            if estop:
                break

        # exit epochs loop
        if estop:
            break

        print 'Seen %d samples' % nsamples
        print("Epoch {} of {} took {:.3f}s".format(
            eidx + 1, max_epochs, time.time() - start_time))

    max_valid_idx = np.argmax(np.array(history_acc)[:, 3])
    best = history_acc[max_valid_idx]
    best = (round(best[0], 5), round(best[3], 5), round(best[6], 5),
            round(best[7], 5), round(best[8], 5))
    print("Global Accuracies :")
    print 'Best: Train ', best[0], 'Valid ', best[1], 'Test ', best[2]
    print("Test Mean Class Accuracy:", best[3])
    print("Test Mean Intersection Over Union:", best[4])

    if len(saveto) != 1:
        print("Moving temporary model files to {}".format(saveto[1]))
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
    color_list = color_list_datasets[name]

    print("Loading data ...")
    load_data, properties = get_dataset(name)
    train, valid, test, mean, std, filenames = load_data(
        resize_images=options['resize_images'],
        resize_size=options['resize_size'],
        color=options['color'],
        color_space=options['color_space'],
        with_filenames=True)

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

    for im, mask, f in zip(images, gt, filenames[id_f]):
        outpath = os.path.join(seg_path, dataset_set, os.path.basename(f))
        mask_rgb = label2rgb(mask, colors=color_list)
        mask_pred = load(outpath)

        fig = plt.figure(figsize=(20, 10))
        a = fig.add_subplot(1, 3, 1)
        imgplot = plt.imshow(im)

        a = fig.add_subplot(1, 3, 2)
        plt.imshow(mask_rgb)

        a = fig.add_subplot(1, 3, 3)
        imgplot = plt.imshow(mask_pred)
        plt.show(block=False)
        raw_input("Press Enter to show the next image")

        plt.close(fig)

    else:
        pass
        # TODO: we need a single function with alla the params to create the
        # model so we call it from train or from show_seg

        # out_layer = ReSeg()
        # # load params
        # with np.load('model.npz') as f:
        #     bestparams_val = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(out_layer, bestparams_val)
        #
        # #compute prediction on the dataset or on the image that we specified

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
