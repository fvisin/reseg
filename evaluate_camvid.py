from reseg import train
import lasagne


def main(job_id, params):
    train_acc, valid_acc, test_acc, test_mean_class_acc, test_mean_iou = train(

        saveto=params['saveto'],
        tmp_saveto=params['tmp-saveto'],
        # Input Conv layers
        in_nfilters=params['in-nfilters'],
        in_filters_size=params['in-filters-size'],
        in_filters_stride=params['in-filters-stride'],
        in_W_init=params['in-W-init'],
        in_b_init=params['in-b-init'],
        in_nonlinearity=params['in-nonlinearity'],

        # RNNs layers
        dim_proj=params['dim-proj'],
        pwidth=params['pwidth'],
        pheight=params['pheight'],
        stack_sublayers=params['stack-sublayers'],
        RecurrentNet=params['RecurrentNet'],
        nonlinearity=params['nonlinearity'],
        hid_init=params['hid-init'],
        grad_clipping=params['grad-clipping'],
        precompute_input=params['precompute-input'],
        mask_input=params['mask-input'],

        # GRU specific params
        gru_resetgate=params['gru-resetgate'],
        gru_updategate=params['gru-updategate'],
        gru_hidden_update=params['gru-hidden-update'],
        gru_hid_init=params['gru-hid-init'],

        # LSTM specific params
        lstm_ingate=params['lstm-ingate'],
        lstm_forgetgate=params['lstm-forgetgate'],
        lstm_cell=params['lstm-cell'],
        lstm_outgate=params['lstm-outgate'],

        # RNN specific params
        rnn_W_in_to_hid=params['rnn-W-in-to-hid'],
        rnn_W_hid_to_hid=params['rnn-W-hid-to-hid'],
        rnn_b=params['rnn-b'],

        # Output upsampling layers
        out_upsampling=params['out-upsampling'],
        out_nfilters=params['out-nfilters'],
        out_filters_size=params['out-filters-size'],
        out_filters_stride=params['out-filters-stride'],
        out_W_init=params['out-W-init'],
        out_b_init=params['out-b-init'],
        out_nonlinearity=params['out-nonlinearity'],

        # Prediction, Softmax
        intermediate_pred=params['intermediate-pred'],
        class_balance=params['class-balance'],

        # Special layers
        batch_norm=params['batch-norm'],
        use_dropout=params['use-dropout'],
        dropout_rate=params['dropout-rate'],
        use_dropout_x=params['use-dropout-x'],
        dropout_x_rate=params['dropout-x-rate'],

        # Optimization method
        optimizer=params['optimizer'],
        learning_rate=params['learning-rate'],
        momentum=params['momentum'],
        rho=params['rho'],
        beta1=params['beta1'],
        beta2=params['beta2'],
        epsilon=params['epsilon'],
        weight_decay=params['weight-decay'],
        weight_noise=params['weight-noise'],

        # Early stopping
        patience=params['patience'],
        max_epochs=params['max-epochs'],
        min_epochs=params['min-epochs'],

        # Sampling and validation params
        validFreq=params['validFreq'],
        saveFreq=params['saveFreq'],
        n_save=params['n-save'],

        # Batch params
        batch_size=params['batch-size'],
        valid_batch_size=params['valid-batch-size'],
        shuffle=params['shuffle'],

        # Dataset
        dataset=params['dataset'],
        color_space=params['color-space'],
        color=params['color'],
        resize_images=params['resize-images'],
        resize_size=params['resize-size'],

        # Pre_processing
        preprocess_type=params['preprocess-type'],
        patch_size=params['patch-size'],
        max_patches=params['max-patches'],

        # Data augmentation
        do_random_flip=params['do-random-flip'],
        do_random_shift=params['do-random-shift'],
        do_random_invert_color=params['do-random-invert-color'],
        shift_pixels=params['shift-pixels'],
        reload_=params['reload']

        # fixed params
    )
    return train_acc, valid_acc, test_acc, test_mean_class_acc, test_mean_iou


if __name__ == '__main__':
    dataset = 'camvid'
    main(1, {
        'saveto':  dataset + '_models/model_recseg_' + dataset + '.npz',
        'tmp-saveto':  'tmp/model_recseg_' + dataset + '.npz',

        # Note: with linear_conv you cannot select every filter size.
        # It is not trivial to invert with expand unless they are a
        # multiple of the image size, i.e., you would have to "blend" together
        # multiple predictions because one pixel cannot be fully predicted just
        # by one element of the last feature map
        # call ConvNet.compute_reasonable_values() to find these
        # note you should pick one pair (p1, p2) from the first list and
        # another pair (p3, p4) from the second, then set in_filter_size
        # to be (p1, p3),(p2, p4)
        # valid: 1 + (input_dim - filter_dim) / stride_dim

        # Input Conv layers
        'in-nfilters': None,  # None = no input convolution
        'in-filters-size': (),
        'in-filters-stride': (),
        'in-W-init': lasagne.init.GlorotUniform(),
        'in-b-init': lasagne.init.Constant(0.),
        'in-nonlinearity': lasagne.nonlinearities.rectify,

        # RNNs layers
        'dim-proj': [100, 100],
        'pwidth': [2, 1],
        'pheight': [2, 1],
        'stack-sublayers': (True, True),
        'RecurrentNet': lasagne.layers.GRULayer,
        'nonlinearity': lasagne.nonlinearities.rectify,
        'hid-init': lasagne.init.Constant(0.),
        'grad-clipping': 0,
        'precompute-input': True,
        'mask-input': None,

        # GRU specific params
        'gru-resetgate': lasagne.layers.Gate(W_cell=None),
        'gru-updategate': lasagne.layers.Gate(W_cell=None),
        'gru-hidden-update': lasagne.layers.Gate(
          W_cell=None,
          nonlinearity=lasagne.nonlinearities.tanh),
        'gru-hid-init': lasagne.init.Constant(0.),

        # LSTM specific params
        'lstm-ingate': lasagne.layers.Gate(),
        'lstm-forgetgate': lasagne.layers.Gate(),
        'lstm-cell': lasagne.layers.Gate(
          W_cell=None,
          nonlinearity=lasagne.nonlinearities.tanh),
        'lstm-outgate': lasagne.layers.Gate(),

        # RNN specific params
        'rnn-W-in-to-hid': lasagne.init.Uniform(),
        'rnn-W-hid-to-hid': lasagne.init.Uniform(),
        'rnn-b': lasagne.init.Constant(0.),

        # Output upsampling layers
        'out-upsampling': 'linear',
        'out-nfilters': None,  # The last number should be the num of classes
        'out-filters-size': (1, 1),
        'out-filters-stride': None,
        'out-W-init': lasagne.init.GlorotUniform(),
        'out-b-init': lasagne.init.Constant(0.),
        'out-nonlinearity': lasagne.nonlinearities.rectify,

        # Prediction, Softmax
        'intermediate-pred': None,
        'class-balance': None,

        # Special layers
        'batch-norm': False,
        'use-dropout': False,
        'dropout-rate': 0.5,
        'use-dropout-x': False,
        'dropout-x-rate': 0.8,

        # Optimization method
        'optimizer': lasagne.updates.adadelta,
        'learning-rate': None,
        'momentum': None,
        'rho': None,
        'beta1': None,
        'beta2': None,
        'epsilon': None,
        'weight-decay': 0.,  # l2 reg
        'weight-noise': 0.,

        # Early stopping
        'patience': 500,  # Num updates with no improvement before early stop
        'max-epochs': 5000,
        'min-epochs': 100,

        # Sampling and validation params
        'validFreq': -1,
        'saveFreq': -1,  # Parameters pickle frequency
        'n-save': -1,  # If n-save is a list of indexes, the corresponding
                       # elements of each split are saved. If n-save is an
                       # integer, n-save random elements for each split are
                       # saved. If n-save is -1, all the dataset is saved
        # Batch params
        'batch-size': 1,
        'valid-batch-size': 1,
        'shuffle': True,

        # Dataset
        'dataset': dataset,
        'color-space': 'RGB',
        'color': True,
        'resize-images': True,
        'resize-size': (240, 320),

        # Pre-processing
        'preprocess-type': None,
        'patch-size': (9, 9),
        'max-patches': 1e5,

        # Data augmentation
        'do-random-flip': False,
        'do-random-shift': False,
        'do-random-invert-color': False,
        'shift-pixels': 2,
        'reload': False
    })
