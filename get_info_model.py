import cPickle as pkl
import matplotlib.pyplot as plt
import numpy
import os
import sys
import collections
from tabulate import tabulate

from config_datasets import headers_datasets

def get_all(in_filename='model_recseg_camvid.npz.pkl', plot=True,
            multiple=False):
    filename = in_filename
    print filename
    if not os.path.isfile(filename):
        return 0
    fp = pkl.load(open(filename, 'rb'))
    print_params(fp, print_commit_hash=not multiple, plot=True,
                 print_history=True)


def print_params(fp, print_commit_hash=False, plot=False,
                 print_history=False):
    """Prints the parameter of the model

    Parameters
    ----------
    fp : dictionary
        The parameters dictionary
    parameter_only : bool
        If True, the commit hash
    plot : blot
        If True, the error curves will be plotted
    """
    dataset = fp.get("dataset", "camvid")
    headers = headers_datasets.get(dataset, None)
    if dataset is None:
        headers = [str(i) for i in range(0, fp['out_nfilters'][-1])]

    errs = numpy.array(fp['history_acc'])
    conf_matrices = numpy.array(fp['history_conf_matrix'])
    iou_indeces = numpy.array(fp['history_iou_index'])

    # they're already accuracies
    if len(errs):
        min_valid = numpy.argmax(errs[:, 3])
        best = errs[min_valid]
        best_test_class_acc = numpy.round(numpy.diagonal(conf_matrices[
                                                                 min_valid][
                                                                 2]), 3)
        best_test_iou_indeces = numpy.round(iou_indeces[min_valid][2], 3)
        if len(best) == 2:
            error = (" ", round(best[0], 6), round(best[3], 6))
        else:
            error = (round(best[0], 6), round(best[3], 6),
                     round(best[6], 6), round(best[7], 6), round(best[8], 6))
    else:
        error = [' ', ' ', ' ', ' ', ' ']
        best_test_class_acc = []

    if 'history_unoptimized_cost' in fp:
        huc = fp['history_unoptimized_cost']
    else:
        huc = None

    print("{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|"
          "{14}|{15}|{16}|{17}|{18}|{19}|{20}|{21}|{22}|{23}|{24}|{25}|"
          "{26}|{27}|{28}|{29}|{30}|{31}|{32}|{33}|{34}|{35}|{36}|{37}|"
          "{38}|{39}|{40}|{41}|{42}|{43}|{44}|{45}|{46}|{47}|{48}|{49}|"
          "{50}|{51}|{52}|{53}|{54}|{55}|{56}|{57}|{58}|{59}|{60}|{61}|"
          "{62}|{63}|{64}|{65}|{66}|").format(
        # Input Conv layers
        fp['in_nfilters'],
        fp['in_filters_size'] if isinstance(fp['in_nfilters'],
                                            collections.Iterable) else ' ',
        fp['in_filters_stride'] if isinstance(fp['in_nfilters'],
                                              collections.Iterable) else ' ',
        fp['in_W_init'].__class__.__name__,
        fp['in_b_init'].__class__.__name__ + ' ' + str(fp['in_b_init'].val),
        fp['in_nonlinearity'].__name__,
        # 0 -> 5

        # RNNs layers
        fp['dim_proj'],
        fp['pwidth'],
        fp['pheight'],
        fp['stack_sublayers'],
        fp['RecurrentNet'].__name__,
        fp['nonlinearity'].__name__,
        fp['hid_init'].__class__.__name__ + ' ' + str(fp['hid_init'].val),
        fp['grad_clipping'],
        # fp['precompute_input'],
        # fp['mask_input'],
        # 6 -> 13

        # GRU specific fp
        fp['gru_resetgate'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'GRULayer' else ' ',

        fp['gru_updategate'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'GRULayer' else ' ',

        fp['gru_hidden_update'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'GRULayer' else ' ',

        fp['gru_hid_init'].__class__.__name__ + ' ' +
                                                str(fp['gru_hid_init'].val)
        if fp['RecurrentNet'].__name__ == 'GRULayer' else ' ',
        # 14 -> 17

        # LSTM specific fp
        fp['lstm_ingate'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'LSTMLayer' else ' ',

        fp['lstm_forgetgate'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'LSTMLayer' else ' ',

        fp['lstm_cell'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'LSTMLayer' else ' ',

        fp['lstm_outgate'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'LSTMLayer' else ' ',
        # 18 -> 21

        # RNN specific fp
        fp['rnn_W_in_to_hid'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'RNNLayer' else ' ',

        fp['rnn_W_hid_to_hid'].__class__.__name__
        if fp['RecurrentNet'].__name__ == 'RNNLayer' else ' ',

        fp['rnn_b'].__class__.__name__ + ' ' + str(fp['rnn_b'].val)
        if fp['RecurrentNet'].__name__ == 'RNNLayer' else ' ',
        # 22 -> 24

        # Output upsampling layers
        fp['out_upsampling'],
        fp['out_nfilters'] if fp['out_upsampling'] == 'grad' else ' ',
        fp['out_filters_size'] if fp['out_upsampling'] == 'grad' else ' ',
        fp['out_filters_stride'] if fp['out_upsampling'] == 'grad' else ' ',
        fp['out_W_init'].__class__.__name__,
        fp['out_b_init'].__class__.__name__ + ' ' + str(fp['out_b_init'].val),
        fp['out_nonlinearity'].__name__,
        # 25 -> 31

        # Prediction, Softmax
        fp['intermediate_pred'],
        fp['class_balance'],
        # 32 -> 33

        # Special layers
        fp['batch_norm'],
        fp['use_dropout'],
        fp['dropout_rate'] if fp['use_dropout'] else ' ',
        fp['use_dropout_x'],
        fp['dropout_x_rate'] if fp['use_dropout_x'] else ' ',
        # 34 -> 38

        # Optimization method
        fp['optimizer'],
        fp['lrate'],
        fp['weight_decay'],
        fp['weight_noise'],
        # 39 -> 42

        # Early stopping
        fp['patience'],
        fp['max_epochs'],
        fp['min_epochs'],
        # 43 -> 45

        # Save, display fp
        # fp['dispFreq'],
        # fp['validFreq'],
        # fp['saveFreq'],
        # fp['n_save'],

        # Batch fp
        fp['batch_size'],
        fp['valid_batch_size'],
        fp['shuffle'],
        # 46 -> 48

        # Dataset
        fp['dataset'],
        fp['color_space'],
        fp['color'],
        fp['resize_images'],
        fp['resize_size'],
        # 49 -> 53

        # Pre_processing
        fp['preprocess_type'],
        fp['patch_size'],
        fp['max_patches'] if fp['preprocess_type'] in ('conv-zca', 'sub-lcn',
                                                       'subdiv-lcn',
                                                       'local_mean_sub')
        else ' ',
        # 54 -> 56

        # Data augmentation
        fp['do_random_flip'],
        fp['do_random_shift'],
        fp['do_random_invert_color'],
        fp['shift_pixels'],
        fp['reload_'],
        error[0],
        error[1],
        error[2],
        error[3],
        error[4]
        # 57 -> 66
    )

    if len(best_test_class_acc) > 0:
        print "|".join(best_test_class_acc.astype(str))

    if 'recseg_git_commit' in fp and print_commit_hash:
        print("Recseg commit: %s" % fp['recseg_git_commit'])

    # plot error curves
    if plot:
        if errs.shape[1] == 2:
            newerrs = numpy.zeros([errs.shape[0], errs.shape[1]+1])
            newerrs[:, 1:3] = errs
            errs = newerrs

        #plt.subplot(2 if huc is not None else 1, 1, 1)

        # Plot Global Pixels % error
        plt.subplot(3, 1, 1)
        plt_range = range(len(errs))
        plt.plot(plt_range, 1 - errs[:, 0], label='train')
        plt.plot(plt_range, 1 - errs[:, 3], label='valid')
        plt.plot(plt_range, 1 - errs[:, 6], label='test')
        plt.grid(True)
        plt.ylim(-0.001, 1.1)
        plt.ylabel('Global Pixels error %')
        plt.legend(loc='best', fancybox=True, framealpha=0.1)

        # plot Mean Pixels error %
        plt.subplot(3, 1, 2)
        plt_range = range(len(errs))
        plt.plot(plt_range, 1 - errs[:, 1], label='train')
        plt.plot(plt_range, 1 - errs[:, 4], label='valid')
        plt.plot(plt_range, 1 - errs[:, 7], label='test')
        plt.grid(True)
        plt.ylim(-0.001, 1.1)
        plt.ylabel('Mean Pixels error %')
        plt.legend(loc='best', fancybox=True, framealpha=0.1)

        # Plot Mean IoU error %
        plt.subplot(3, 1, 3)
        plt_range = range(len(errs))
        plt.plot(plt_range, 1 - errs[:, 2], label='train')
        plt.plot(plt_range, 1 - errs[:, 5], label='valid')
        plt.plot(plt_range, 1 - errs[:, 8], label='test')
        plt.grid(True)
        plt.ylim(-0.001, 1.1)
        plt.ylabel('Mean IoU error %')
        plt.legend(loc='best', fancybox=True, framealpha=0.1)


        if huc is not None:
            plt.subplot(2, 1, 2)
            scale = float(len(errs)) / len(huc)
            huc_range = [i * scale for i in range(len(huc))]
            plt.plot(huc_range, huc)
            plt.ylabel('Training cost')
            plt.grid(True)
        plt.show()
    if print_history:
        for e, c, iou in zip(errs, conf_matrices, iou_indeces):

            train_global_acc, \
             train_mean_class_acc, \
             train_mean_iou_index, \
             valid_global_acc, \
             valid_mean_class_acc, \
             valid_mean_iou_index, \
             test_global_acc, \
             test_mean_class_acc, \
             test_mean_iou_index = e

            train_conf_matrix_norm, \
             valid_conf_matrix_norm, \
             test_conf_matrix_norm, \
             train_conf_matrix, \
             valid_conf_matrix, \
             test_conf_matrix = c

            train_iou_index, \
             valid_iou_index,\
             test_iou_index = iou
            print ""
            print ""
            print ""
            print ""

            headers_acc = ["Global Accuracies",
                           "Mean Class Accuracies",
                           "Mean Intersection Over Union"]

            rows = list()
            rows.append(['Train ',
                        round(train_global_acc, 6),
                        round(train_mean_class_acc, 6),
                        round(train_mean_iou_index, 6)])

            rows.append(['Valid ',
                        round(valid_global_acc, 6),
                        round(valid_mean_class_acc, 6),
                        round(valid_mean_iou_index, 6)])

            rows.append(['Test ', round(test_global_acc, 6),
                         round(test_mean_class_acc, 6),
                         round(test_mean_iou_index, 6)])

            print(tabulate(rows, headers=headers_acc))

            class_acc = list()
            class_acc.append(numpy.concatenate([["Train"], numpy.round(
                numpy.diagonal(train_conf_matrix_norm), 3)]))
            class_acc.append(numpy.concatenate([["Valid"], numpy.round(
                numpy.diagonal(valid_conf_matrix_norm), 3)]))
            if len(test_conf_matrix) > 0:
                class_acc.append(numpy.concatenate([["Test"], numpy.round(
                    numpy.diagonal(test_conf_matrix_norm), 3)]))

            print(tabulate(class_acc, headers=headers))

            numpy.set_printoptions(precision=3)
            print ""
            print('Train Confusion matrix')
            print(tabulate(train_conf_matrix_norm, headers=headers))
            print ""
            print('Valid Confusion matrix')
            print(tabulate(valid_conf_matrix_norm, headers=headers))

            if len(test_conf_matrix_norm) > 0:
                print ""
                print('Test Confusion matrix')
                print(tabulate(test_conf_matrix_norm, headers=headers))

    return 1


if __name__ == '__main__':
    if len(sys.argv) == 3:
        name = sys.argv[1]
        n = sys.argv[2]
        get_all(name + '_models/model_recseg_' + name + n + '.npz.pkl')
    else:
        if len(sys.argv) == 2:
            name = sys.argv[1]
        else:
            name = 'camvid'
        n = 0
        ok = 1
        while ok:
            n += 1
            if n == 23:
                continue
            if n in [39, 40, 45, 48]:
                print ''
                continue
            ok = get_all('model_recseg_' + name + str(n) + '.npz.pkl', False,
                         True)
            if not ok:
                ok = get_all('/Tmp/visin/model_recseg_' + name + str(n) +
                             '.npz.pkl', False, True)
            if not ok:
                ok = get_all('camvid_models/model_recseg_' + name + str(n) +
                             '.npz.pkl', False, True)
        print('Printed models from 1 to {}').format(n-1)
