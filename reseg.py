import time
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from layers import buildReSeg

from utils import *
from get_info_model import print_params
from subprocess import check_output
from skimage.data import load
from skimage.color import label2rgb
from config_datasets import color_list_datasets
import matplotlib.pyplot as plt

# DATASET IMPORT
import camvid
import daimler
import kitti_road

floatX = theano.config.floatX


# Number of training sequences in each batch
# batch_size = 1

# Optimization learning rate
# LEARNING_RATE = .0001

# All gradients above this will be clipped
# GRAD_CLIP = 100

# Number of epochs to train the net
# NUM_EPOCHS = 100

# ReNet Layers
# n_layers = 2

# Patch size
# PWIDTH = (2, 2)
# PHEIGHT = (2, 2)

# Number of units in the hidden (recurrent) layer
# dim_proj = (100, 100)


# stack_sublayers = (False, False)
# saveto = "recseg_lasagne_camvid.npz"
datasets = {'camvid': (camvid.load_data, camvid.properties),
            'daimler': (daimler.load_data, daimler.properties),
            'kitti_road': (kitti_road.load_data, kitti_road.properties)}


def get_dataset(name):
    return (datasets[name][0], datasets[name][1])


def train(
        saveto='model_resec.npz',
        # tmp_saveto=None,
        dataset='camvid',
        # in_nfilters=None,  # None = no input convolution
        # in_filters_size=[],
        # in_filters_stride=[],
        # in_init='glorot',
        # in_activ='tanh',
        encoder='gru',
        # intermediate_pred=None,
        dim_proj=(50, 50),
        pwidth=(2, 2),
        pheight=(2, 2),
        stack_sublayers=(True, True),
        out_upsampling='deconvnet',
        out_nfilters=(100, 12),  # The last number should be the num of classes
        out_filters_size=((2, 2), (2, 2)),
        out_filters_stride=((2, 2), (2, 2)),
        out_init='glorot',
        out_activ='identity',
        optimizer='adadelta',
        weight_decay=0.,  # l2 reg
        weight_noise=0.,
        lrate=0.01,
        clip_grad_threshold=0.,
        patience=50,
        max_epochs=5000,
        dispFreq=100,
        validFreq=-1,
        saveFreq=-1,  # parameters pickle frequency
        batch_size=8,
        valid_batch_size=1,
        # use_dropout=False,
        # dropout_rate=0.5,
        # use_dropout_x=False,
        # dropout_x_rate=0.8,
        color_space='RGB',
        color=True,
        preprocess_type=None,
        # patch_size=(9, 9),
        # max_patches=1e5,
        class_balance=None,
        # batch_norm=False,
        rnn_init='norm_weight',
        rnn_activation='tanh',
        shuffle=True,
        # do_random_flip=False,
        # do_random_shift=False,
        # do_random_invert_color=False,
        # shift_pixels=2,
        reload_=False,
        resize_images=True,
        resize_size=-1,
        # mapping_labels=None
        ):

    # Set options and history_errs
    # start = time.time()  # use time() to know the actual real-world time
    # best_p = {}
    # suppress_dropout = theano.shared(np.float32(0.))
    trng = RandomStreams(0xbeef)
    rng = np.random.RandomState(0xbeef)
    # saveto = [tmp_saveto, saveto] if tmp_saveto else [saveto]
    if type(pwidth) != list:
        pwidth = [pwidth] * len(dim_proj)
    if type(pheight) != list:
        pheight = [pheight] * len(dim_proj)
    # if intermediate_pred is None:
    #     intermediate_pred = [[False] * (len(dim_proj) - 1)] + [[False, True]]
    # if not unroll(intermediate_pred)[-1]:
    #     raise ValueError('The last value of intermediate_pred should be True')

    if not resize_images and valid_batch_size != 1:
        raise ValueError('When images are not resized valid_batch_size'
                         'should be 1')
    color = color if color else False
    nchannels = 3 if color else 1
    mode = None
    reorder = None
    options = locals().copy()
    options['recseg_git_commit'] = check_output(
        'git rev-parse HEAD', shell=True)
    options['trng'] = [el[0].get_value() for el in trng.state_updates]
    options['history_errs'] = np.array([])
    options['history_conf_matrix'] = np.array([])
    options['history_iou_index'] = np.array([])
    if reload_:
        for s in saveto[::-1]:
            try:
                with open('%s.pkl' % s, 'rb') as f:
                    options_reloaded = pkl.load(f)
                    for k, v in options.iteritems():
                        if k in ['trng', 'history_errs',
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
    dataset = options['dataset']
    # in_nfilters = options['in_nfilters']
    # in_filters_size = options['in_filters_size']
    # in_filters_stride = options['in_filters_stride']
    # in_init = options['in_init'] if 'in_init' in options else 'glorot'
    # in_activ = options['in_activ'] if 'in_activ' in options else 'identity'
    encoder = options['encoder']
    # intermediate_pred = options['intermediate_pred']
    dim_proj = options['dim_proj']
    pwidth = options['pwidth']
    pheight = options['pheight']
    out_upsampling = options['out_upsampling']
    out_nfilters = options['out_nfilters']
    out_filters_size = options['out_filters_size']
    out_filters_stride = options['out_filters_stride']
    out_init = options['out_init']
    out_activ = options['out_activ']
    optimizer = options['optimizer']
    weight_decay = options['weight_decay']
    weight_noise = options['weight_noise']
    lrate = options['lrate']
    clip_grad_threshold = options['clip_grad_threshold']
    batch_size = options['batch_size']
    valid_batch_size = options['valid_batch_size']
    # use_dropout = options['use_dropout']
    # dropout_rate = options['dropout_rate']
    # use_dropout_x = options['use_dropout_x']
    # dropout_x_rate = options['dropout_x_rate']
    color_space = options['color_space']
    color = options['color']
    preprocess_type = options['preprocess_type']
    # patch_size = options['patch_size']
    # max_patches = options['max_patches']
    class_balance = options['class_balance']
    # batch_norm = options['batch_norm']
    rnn_init = options['rnn_init']
    rnn_activation = options['rnn_activation']
    shuffle = options['shuffle']
    # reorder, random_flip/shift/invert_colors, shift pixels
    resize_images = options['resize_images']
    resize_size = options['resize_size']
    rng = options['rng']
    # trng = options['trng'] --> to be reloaded after building the model
    history_errs = options['history_errs'].tolist()
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

    print("Loading data ...")
    load_data, properties = get_dataset(dataset)
    train, valid, test, mean, std, filenames = load_data(
        resize_images=resize_images,
        resize_size=resize_size,
        color=color,
        color_space=color_space,
        rng=rng,
        with_filenames=True)
    nclasses = max([np.max(el) for el in train[1]]) + 1
    x_train, y_train = train

    w_freq = 1
    if class_balance in ['median_freq_cost', 'rare_freq_cost']:
        u_train, c_train = np.unique(train[1], return_counts=True)
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

    # How often should we check the output?
    # Now we want to check at each epoch so we check after
    # batch_size minibatchs train iterations
    
    if validFreq == -1:
        validFreq = len(x_train)
    if saveFreq == -1:
        saveFreq = len(x_train) 

    filenames_train, filenames_valid, filenames_test = filenames
    cheight, cwidth, cchannels = x_train[0].shape

    # TODO: we can use the preprocess function here

    print("Building model ...")

    input_shape = (batch_size, cheight, cwidth, cchannels)
    
    out_layer, f_pred, f_train = buildReSeg(input_shape,
                                            n_layers,
                                            pheight,
                                            pwidth,
                                            dim_proj,
                                            nclasses,
                                            stack_sublayers,
                                            out_upsampling,
                                            out_nfilters,
                                            out_filters_size,
                                            out_filters_stride)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    idx = 0
    for epoch in range(max_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        traibatch_sizees = 0
        start_time = time.time()
        for minibatch in iterate_minibatches(x_train.astype(floatX),
                                             y_train,
                                             batch_size,
                                             shuffle=shuffle):
            inputs, targets, _ = minibatch
            targets_flat = targets.flatten()
            w = 1
            if class_balance in ['median_freq_cost', 'rare_freq_cost']:
                w = np.sum(w_freq[targets_flat]).astype(floatX)

            err = f_train(inputs, targets_flat, w)

            if idx % dispFreq == 0:
                print("Iteration {} : ".format(idx))
                print("Minibatch input shape: {} , error {}".format(
                        inputs.shape,
                        err))
            train_err += err
            traibatch_sizees += 1
            idx += 1

            # validate the model each validFreq minibatch iterations
            # if validFreq == -1 then you validate at the end
            # of each epoch
            if idx % validFreq == 0:
                (train_global_acc,
                 train_conf_matrix,
                 train_conf_matrix_norm,
                 train_mean_class_acc,
                 train_iou_index,
                 train_mean_iou_index) = validate(f_pred, train,
                                                  valid_batch_size,
                                                  nclasses,
                                                  filenames=filenames_train,
                                                  folder_dataset='train',
                                                  dataset=dataset,
                                                  saveto=saveto,
                                                  )

                (valid_global_acc,
                 valid_conf_matrix,
                 valid_conf_matrix_norm,
                 valid_mean_class_acc,
                 valid_iou_index,
                 valid_mean_iou_index) = validate(f_pred, valid,
                                                  valid_batch_size,
                                                  nclasses,
                                                  filenames=filenames_valid,
                                                  folder_dataset='valid',
                                                  dataset=dataset,
                                                  saveto=saveto,
                                                  )

                (test_global_acc,
                 test_conf_matrix,
                 test_conf_matrix_norm,
                 test_mean_class_acc,
                 test_iou_index,
                 test_mean_iou_index) = validate(f_pred, test,
                                                 valid_batch_size,
                                                 nclasses,
                                                 filenames=filenames_test,
                                                 folder_dataset='test',
                                                 dataset=dataset,
                                                 saveto=saveto,
                                                 )

                # TODO:
                # here we can also use other performance index to early stop
                # e.g Mean IoU, Mean Class Accuracy
                # we can think of a parameter to decide

                valid_err = 1 - valid_global_acc
                # save the best params
                if (len(history_errs) == 0 or valid_err <= 1 - np.array(
                        history_errs)[:, 3].min()):
                    np.savez(saveto, *lasagne.layers.get_all_param_values(
                                     out_layer))

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

                history_errs.append([train_global_acc,
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
                                           test_conf_matrix,
                                            ])

                history_iou_index.append([train_iou_index,
                                         valid_iou_index,
                                         test_iou_index])
            if idx % saveFreq == 0:
                options['history_errs'] = np.array(history_errs)
                options['history_conf_matrix'] = np.array(history_conf_matrix)
                options['history_iou_index'] = np.array(history_iou_index)

                print "Saving the options to {}".format(saveto)
                pkl.dump(options,
                         open('%s.pkl' % saveto, 'wb'))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - start_time))

    # After training, we compute and print the test error:
    (train_global_acc,
     train_conf_matrix,
     train_conf_matrix_norm,
     train_mean_class_acc,
     train_iou_index,
     train_mean_iou_index) = validate(f_pred, train,
                                      valid_batch_size,
                                      nclasses,
                                      filenames=filenames_train,
                                      folder_dataset='train',
                                      dataset=dataset,
                                      saveto=saveto,
                                      )

    (valid_global_acc,
     valid_conf_matrix,
     valid_conf_matrix_norm,
     valid_mean_class_acc,
     valid_iou_index,
     valid_mean_iou_index) = validate(f_pred, valid,
                                      valid_batch_size,
                                      nclasses,
                                      filenames=filenames_valid,
                                      folder_dataset='valid',
                                      dataset=dataset,
                                      saveto=saveto,
                                      )
    (test_global_acc,
     test_conf_matrix,
     test_conf_matrix_norm,
     test_mean_class_acc,
     test_iou_index,
     test_mean_iou_index) = validate(f_pred, test,
                                     batch_size, nclasses,
                                     filenames=filenames_test,
                                     folder_dataset='test',
                                     saveto=saveto,
                                     )

    print("")
    print("Global Accuracies :")
    print('Test ', test_global_acc)
    print("")
    print("Class Accuracies :")
    print('Test ', test_mean_class_acc)
    print("")
    print("Mean Intersection Over Union :")
    print('Test ', test_mean_iou_index)
    print("")

    # Optionally, you could now dump the network weights to a file like this:

    # TODO : save only the best params, not the last ones!!
    # if saveto[:-4] != ".npz":
    #     saveto += ".npz"
    # np.savez(saveto, *lasagne.layers.get_all_param_values(
    #         out_layer))

    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return (train_global_acc, valid_global_acc, test_global_acc,
            test_mean_class_acc, test_mean_iou_index)


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
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(out_layer, param_values)
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




