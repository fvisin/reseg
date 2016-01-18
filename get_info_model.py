import cPickle as pkl
import matplotlib.pyplot as plt
import numpy
import os
import sys
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

    errs = numpy.array(fp['history_errs'])
    conf_matrices = numpy.array(fp['history_conf_matrix'])
    iou_indeces = numpy.array(fp['history_iou_index'])

    # they're already accuracies
    if len(errs):
        min_valid = numpy.argmax(errs[:, 8])
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

    if 'encoder' in fp:
        print("{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|"
              "{14}|{15}|{16}|{17}|{18}|{19}|{20}|{21}|{22}|{23}|{24}|{25}|"
              "{26}|{27}|{28}|{29}|{30}|{31}|{32}|{33}|{34}|").format(
            fp['encoder'],
            fp['optimizer'],
            fp['color'],
            fp['color_space'] if 'color_space' in fp else 'RGB',
            fp['preprocess_type'] if 'preprocess_type' in fp else ' ',
            fp['class_balance'] if 'class_balance' in fp else ' ',
            fp['resize_images'],
            fp['resize_size'] if 'resize_size' in fp else ' ',
            # 7 --> 11
            fp['do_random_flip']if 'do_random_flip' in fp else 'False',
            fp['do_random_shift'] if 'do_random_shift' in fp else 'False',
            fp['do_random_invert_color'] if 'do_random_invert_color' in fp else 'False',
            [fp['use_dropout_x'], fp['use_dropout']] if 'use_dropout' in fp and
                                                        'use_dropout_x' in fp
            else [' ', ' '],
            fp['dropout_x_rate'] if 'dropout_x_rate' in fp else ' ',
            # 12 --> 16
            fp['lrate'],
            fp['weight_decay'] if 'weight_decay' in fp else fp['decay_c'],
            fp['clip-grad-threshold'] if 'clip-grad-threshold' in fp else ' ',
            fp['batch_size'],
            fp['in_nfilters'] if 'in_nfilters' in fp else ' ',
            # 17 --> 21
            fp['in_filters_size'] if 'in_filters_size' in fp else ' ',
            fp['in_filters_stride'] if 'in_filters_stride' in fp else ' ',
            fp['in_activ'] if 'in_activ' in fp else ' ',
            [fp['pheight'], fp['pwidth']],
            fp['dim_proj'],
            # 22 --> 25
            fp['intermediate_pred'] if 'intermediate_pred' in fp else ' ',
            fp['stack_sublayers'] if 'stack_sublayers' in fp else ' ',
            fp['out_upsampling'] if 'out_upsampling' in fp else ' ',
            fp['out_nfilters'] if 'out_nfilters' in fp else ' ',
            fp['out_filters_size'] if 'out_filters_size' in fp else ' ',
            fp['out_filters_stride'] if 'out_filters_stride' in fp else ' ',
            # 27 --> 30
            fp['out_activ'] if 'out_activ' in fp else ' ',
            error[0],
            error[1],
            error[2],
            error[3],
            error[4])
    else:
        print("{0}|{1}|{2}|{3}|{4}".format(error[0],
                                           error[1],
                                           error[2],
                                           error[3],
                                           error[4]))

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