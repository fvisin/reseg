from reseg import train


def main(job_id, params):
    print('Anything printed here will end up in the output directory for job'
          '#%d' % job_id)
    print params
    train_acc, valid_acc, test_acc, test_mean_class_acc, test_mean_iou = train(
        saveto=params['saveto'],
        tmp_saveto=params['tmp-saveto'],
        in_nfilters=params['in-nfilters'],
        in_filters_size=params['in-filters-size'],
        in_filters_stride=params['in-filters-stride'],
        encoder=params['encoder'],
        # intermediate_pred=params['intermediate-pred'],
        dim_proj=params['dim-proj'],
        pwidth=params['pwidth'],
        pheight=params['pheight'],
        stack_sublayers=params['stack-sublayers'],
        out_upsampling=params['out_upsampling'],
        out_nfilters=params['out-nfilters'],
        out_filters_size=params['out-filters-size'],
        out_filters_stride=params['out-filters-stride'],
        optimizer=params['optimizer'],
        weight_decay=params['weight-decay'],
        weight_noise=params['weight-noise'],
        lrate=params['lrate'],
        clip_grad_threshold=params['clip-grad-threshold'],
        batch_size=params['batch-size'],
        # use_dropout=params['use-dropout'],
        # dropout_rate=params['dropout-rate'],
        # use_dropout_x=params['use-dropout-x'],
        # dropout_x_rate=params['dropout-x-rate'],
        color=params['color'],
        color_space=params['color-space'],
        preprocess_type=params['preprocess-type'],
        # patch_size=params['patch-size'],
        # max_patches=params['max-patches'],
        class_balance=params['class-balance'],
        shuffle=params['shuffle'],
        reload_=params['reload'],
        resize_images=params['resize-images'],
        resize_size=params['resize-size'],
        n_save=params['n-save'],
        # fixed params
        # in_init='glorot',
        # out_init='glorot',
        # in_activ='rectifier',
        # out_activ='rectifier',
        patience=50,
        max_epochs=2,
        dispFreq=1,
        validFreq=-1,
        saveFreq=-1,
        valid_batch_size=params['batch-size'],  # same as batch_size
        dataset='camvid',
        # do_random_flip=True,
        # do_random_shift=True,
        # do_random_invert_color=False,
        # shift_pixels=2
    )
    return train_acc, valid_acc, test_acc, test_mean_class_acc, test_mean_iou

if __name__ == '__main__':
    main(1, {
        'saveto': 'camvid_models/model_recseg' + __file__[8:-3] + '.npz',
        'tmp-saveto': 'tmp/model_recseg' + __file__[8:-3] + '.npz',

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

        'in-nfilters': None,  # None = disable in convolution
        'in-filters-size': [(5, 5), (5, 5)],
        'in-filters-stride': [(4, 4), (1, 1)],
        'encoder': 'gru',
        # 'intermediate-pred': [[False], [True]],
        'dim-proj': [100, 100],
        'pwidth': [2, 1],
        'pheight': [2, 1],
        'stack-sublayers': (True, True),
        'out_upsampling': 'linear',
        # The last number should be the num of classes
        'out-nfilters': (12,),
        'out-filters-size': [(5, 5)],
        'out-filters-stride': [(2, 2)],
        'optimizer': 'adadelta',
        'weight-decay': 0.,
        'weight-noise': 0.,
        'lrate': 0.0001,
        'clip-grad-threshold': 0.,  # 0 = disabled
        'batch-size': 1,
        # 'use-dropout': False,
        # 'dropout-rate': 0.5,
        # 'use-dropout-x': False,
        # 'dropout-x-rate': 0.2,
        'color': True,
        'color-space': 'RGB',
        'preprocess-type': None,
        # 'patch-size': (9, 9),
        # 'max-patches': 1e5,
        'class-balance': None,
        'n-save': 20,
        'shuffle': True,
        'resize-images': True,
        'resize-size': (320, 240),  # ATTENTION: w x h (PIL)!! 264x212
        'reload': False
    })
