import os
import pickle

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes
from scipy import stats
from scipy.optimize import curve_fit


def plot_all_loss_acc(args):
    '''Plot weight difference of this hyperparameter setting.
    '''
    test_loss = []
    test_acc = []
    best_val_acc = []
    best_val_loss = []
    weight_decay_values = [0, 0.00001, 0.00005, 0.00009,
                           0.00015, 0.0003, 0.0005, 0.001]
    # relu+Adam
    # weight_decay_values = [0, 0.000001, 0.00001, 0.00004, 0.0001,
    #                        0.0003, 0.0005, 0.001]
    for weight_decay in weight_decay_values:
        scores = read_scores(args, weight_decay)
        test_loss_temp = np.array([scores[num_repeat][1]
                                   for num_repeat in range(args.num_repeats)])
        test_loss.append(test_loss_temp)
        test_acc_temp = np.array([scores[num_repeat][2]
                                  for num_repeat in range(args.num_repeats)])
        test_acc.append(test_acc_temp)

        print('weight decay: {}, test acc is {}'.format(
            weight_decay, test_acc_temp))

        losses = read_losses(args, weight_decay)
        best_val_loss.append(np.array([np.array([losses[num_repeat]['val'][epochs][1] for epochs in range(args.epochs)]).min()
                                       for num_repeat in range(args.num_repeats)]))
        best_val_acc.append(np.array([np.array([losses[num_repeat]['val'][epochs][2] for epochs in range(args.epochs)]).max()
                                      for num_repeat in range(args.num_repeats)]))
        # losses = read_losses(args, weight_decay)
        # val_acc = [losses['val'][epoch][1] for epoch in range(args.epochs)]
        # best_val_acc.append(np.amax(val_acc))
        # val_loss = [losses['val'][epoch][0] for epoch in range(args.epochs)]
        # best_val_loss.append(np.amin(val_loss))

    # plot test accuracy
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    plt.axvline(x=0.00009, linestyle='--', color='k',
                label=r'$\lambda = {:.5f}$'.format(0.00009))
    plt.errorbar(weight_decay_values, np.array(test_acc).mean(axis=1), yerr=np.array(test_acc).std(axis=1),
                 fmt='o-', markersize=6.5, capsize=2, color='C0', label='Test accuracy', zorder=5)
    ax.locator_params(axis='y', nbins=5)
    ax.set_xscale('symlog', linthreshx=0.0003)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Weight decay strength $\lambda$', fontsize=24)
    plt.ylabel('Test accuracy', fontsize=24)
    # plt.xlim(left=-1e-5)
    plt.legend(fontsize=17.5, loc='lower left',
               bbox_to_anchor=(0.22, 0.0))

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/capacity_test_acc'.format(args.arguments,
                                                         args.num_repeats), dpi=300)

    # plot best validate accuracy
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    plt.axvline(x=0.00009, linestyle='--', color='k',
                label=r'$\lambda = {:.5f}$'.format(0.00009))
    plt.errorbar(weight_decay_values, np.array(best_val_acc).mean(axis=1), np.array(best_val_acc).std(axis=1),
                 fmt='o-', markersize=6.5, capsize=2, color='C0', label=r'Test accuracy$^*$')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xscale('symlog', linthreshx=0.0003)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Weight decay strength $\lambda$', fontsize=24)
    plt.ylabel(r'Test accuracy$^*$', fontsize=24)
    # plt.ylim(0.2, 1)
    plt.legend(fontsize=17.5, loc='lower left',
               bbox_to_anchor=(0.22, 0.0))

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/capacity_val_acc'.format(args.arguments,
                                                        args.num_repeats), dpi=300)

    # plot test loss
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plt.axvline(x=0.00009, linestyle='--', color='k', label='Edge of Chaos')
    plt.errorbar(weight_decay_values, np.array(test_loss).mean(axis=1), yerr=np.array(test_loss).std(axis=1),
                 fmt='o-', markersize=3, capsize=2, color='C2', label='Test loss')

    ax.locator_params(axis='x', nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Weight decay strength', fontsize=24)
    plt.ylabel('Test loss', fontsize=24)
    # plt.ylim(0.2, 1)
    plt.legend(fontsize=22)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/capacity_test_loss'.format(args.arguments,
                                                          args.num_repeats), dpi=300)

    # plot best validate loss
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plt.errorbar(weight_decay_values, np.array(best_val_loss).mean(axis=1), np.array(best_val_loss).std(axis=1),
                 fmt='o-', markersize=3, capsize=2, color='C2', label='Best validate loss')
    plt.axvline(x=0.00009, linestyle='--', color='k', label='Edge of Chaos')
    ax.locator_params(axis='x', nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Weight decay strength', fontsize=24)
    plt.ylabel('Best validate loss', fontsize=24)
    # plt.ylim(0.2, 1)
    plt.legend(fontsize=22)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/capacity_val_loss'.format(args.arguments,
                                                         args.num_repeats), dpi=300)


def plot_weight_decay_path(args):
    '''Plot the combined paths and losses for different weight decay.
    '''
    # Plot model evolution path
    fig, ax = plt.subplots(figsize=(13, 6.5))
    betas, J0s,  separation1_log = prepare_xy(args)
    # Plot separation
    ax = sns.heatmap(separation1_log, xticklabels=J0s,
                     yticklabels=betas[::-1], cmap='Purples')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, rotation=0)
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24)
    ax.set_ylabel(r'$1/\ J$', fontsize=24)
    # Plot colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(
        'Asymptotic distance $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', fontsize=24)
    if args.activation_func == 'relu':
        cbar_labels1 = [-4, -2, 0, 2]
        cbar_labels2 = ['< {}'.format(.0001), str(
            0.01), str(1.0), '>= {}'.format(100)]
    elif args.activation_func == 'tanh':
        cbar_labels1 = [-4, -2, 0]
        cbar_labels2 = ['< {}'.format(.0001), str(0.01), '>= {}'.format(1)]
    cbar.set_ticks(cbar_labels1)
    cbar.set_ticklabels(cbar_labels2)
    cbar.ax.tick_params(labelsize=24)
    # Plot theoretical boundary
    if args.activation_func == 'relu':
        # Prepare boundary data to plot
        mat = spio.loadmat(
            'rawdata/data_100bins_relu.mat', squeeze_me=True)
        integrals = mat['integrals'][::-1]
        # integrals = integrals.asty
        for i in range(0, integrals.shape[1], 2):
            integrals[:, i] = 0
        mask = (integrals - 1) * (-1)
        sns.heatmap(integrals, xticklabels=J0s,
                    yticklabels=betas[::-1], mask=mask, cmap='gray', cbar=False)
    elif args.activation_func == 'tanh':
        plt.axhline(y=(100.5-1*33.3333), color='black', linestyle='--', lw=3)
    # Plot paths
    for weight_decay, color in [(0, 'C1'), (0.00009, 'C3'), (0.0005, 'C0')]:
        results = read_weights(args, weight_decay)
        losses = read_losses(args, weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(args.epochs//1)] for num_repeat in range(1)])
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(args.epochs//1)] for num_repeat in range(1)])
        val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                             for epoch in range(args.epochs//1)] for num_repeat in range(1)])

        J = (np.std(W0s.mean(axis=0), axis=(1, 2))*np.sqrt(args.input_shape))
        J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape

        if args.activation_func == 'relu':
            x, y = (J0/J)*20+100.5, 100.5-(1/J)*50
            plt.plot(x, y, 'o-', color=color,
                     markersize=3, lw=2)
            if weight_decay == 0:
                plt.scatter(x[-1], y[-1], s=300, marker='s', color=color,
                            label='Epoch {} $(\lambda = {})$'.format(args.epochs-1, weight_decay))
            elif weight_decay == 0.00009:
                plt.scatter(x[-1], y[-1], s=300, marker='s', color=color,
                            label='Epoch {} $(\lambda = {:.5f})$'.format(args.epochs-1, weight_decay))
            else:
                plt.scatter(x[-1], y[-1], s=300, marker='s', color=color,
                            label='Epoch {} $(\lambda = {:.4f})$'.format(args.epochs-1, weight_decay))
        elif args.activation_func == 'tanh':
            x, y = (J0/J)*50+50.5, 100.5-(1/J)*33.3333
            plt.plot(x, y, 'o-', color=color, markersize=3, lw=2)
            if weight_decay == 0:
                plt.scatter(x[-1], y[-1], s=300, marker='s', color=color,
                            label='Epoch {} $(\lambda = {})$'.format(args.epochs-1, weight_decay))
            elif weight_decay == 0.00009:
                plt.scatter(x[-1], y[-1], s=300, marker='s', color=color,
                            label='Epoch {} $(\lambda = {:.5f})$'.format(args.epochs-1, weight_decay))
            else:
                plt.scatter(x[-1], y[-1], s=300, marker='s', color=color,
                            label='Epoch {} $(\lambda = {:.4f})$'.format(args.epochs-1, weight_decay))
        # Plot arrow
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        plt.quiver(pos_x[0], pos_y[0], (u/norm)[0], (v/norm)[0], width=0.002, lw=0.5,
                   scale=20, headwidth=20, headlength=20, headaxislength=16, angles="xy", pivot="mid", color=color)
    # Manually add a legend
    handles, labels = ax.get_legend_handles_labels()
    lines = Line2D([0], [0], color='black', linewidth=3, linestyle='--')
    label = 'Edge of chaos ( $J=1$ )'
    handles.insert(0, lines)
    labels.insert(0, label)
    plt.legend(handles=handles, labels=labels,
               fontsize=17.6, loc='upper left')
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24)
    ax.set_ylabel(r'$1/\ J$', fontsize=24)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs(
            'results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/3paths'.format(args.arguments,
                                              args.num_repeats), dpi=300)


def plot_weight_decay_var_mean(args):
    # plot 3 variances
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for weight_decay, color in [(0, 'C1'), (0.00009, 'C3'), (0.0005, 'C0')]:
        results = read_weights(args, weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(args.epochs//1)] for num_repeat in range(1)])

        J = (np.std(W0s.mean(axis=0), axis=(1, 2))*np.sqrt(args.input_shape))
        J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape
        J_squard = np.power(J, 2)
        x = np.arange(args.epochs)
        if weight_decay == 0:
            ax.plot(x[::2], J_squard[::2], 'o-', markersize=3, color=color,
                    label='$\lambda = {}$'.format(weight_decay))
        elif weight_decay == 0.00009:
            ax.plot(x[::2], J_squard[::2], 'o-', markersize=3, color=color,
                    label='$\lambda = {:.5f}$'.format(weight_decay))
        else:
            ax.plot(x[::2], J_squard[::2], 'o-', markersize=3, color=color,
                    label='$\lambda = {:.4f}$'.format(weight_decay))
    plt.axhline(y=1, linestyle='--', color='k', label=r'$J^2 = 1$')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('$J^2$', fontsize=24)
    plt.legend(fontsize=18, loc='lower left', bbox_to_anchor=(0.48, 0.28))

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs(
            'results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/3variance'.format(args.arguments,
                                                 args.num_repeats), dpi=300)

    # plot 3 means
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for weight_decay, color in [(0, 'C1'), (0.00009, 'C3'), (0.0005, 'C0')]:
        results = read_weights(args, weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(args.epochs//1)] for num_repeat in range(1)])

        J = (np.std(W0s.mean(axis=0), axis=(1, 2))*np.sqrt(args.input_shape))
        J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape
        J_squard = np.power(J, 2)
        x = np.arange(args.epochs)
        if weight_decay == 0:
            ax.plot(x[::2], J0[::2], 'o-', markersize=3, color=color,
                    label='$\lambda = {}$'.format(weight_decay))
        elif weight_decay == 0.00009:
            ax.plot(x[::2], J0[::2], 'o-', markersize=3, color=color,
                    label='$\lambda = {:.5f}$'.format(weight_decay))
        else:
            ax.plot(x[::2], J0[::2], 'o-', markersize=3, color=color,
                    label='$\lambda = {:.4f}$'.format(weight_decay))
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('$J_0$', fontsize=24)
    plt.legend(fontsize=18)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs(
            'results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/3mean'.format(args.arguments,
                                             args.num_repeats), dpi=300)

    # plot 2 variances
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for weight_decay, color in [(0.00009, 'C3'), (0.0005, 'C0')]:
        results = read_weights(args, weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(args.epochs//1)] for num_repeat in range(1)])

        J = (np.std(W0s.mean(axis=0), axis=(1, 2))*np.sqrt(args.input_shape))
        J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape
        J_squard = np.power(J, 2)
        x = np.arange(args.epochs)
        ax.plot(x[::2], J_squard[::2], 'o', markersize=3, color=color,
                label='$\lambda = {:.5f}$'.format(weight_decay))
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.locator_params(axis='y', nbins=4)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('$J^2$', fontsize=24)
    plt.legend(fontsize=22)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs(
            'results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/2variance'.format(args.arguments,
                                                 args.num_repeats), dpi=300)

    # plot 2 means
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for weight_decay, color in [(0.00009, 'C3'), (0.0005, 'C0')]:
        results = read_weights(args, weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(args.epochs//1)] for num_repeat in range(1)])

        J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape
        x = np.arange(args.epochs)
        ax.plot(x[::2], J0[::2], 'o', markersize=3, color=color,
                label='$\lambda = {:.4f}$'.format(weight_decay))
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('$J_0$', fontsize=24)
    plt.ylim(top=1)
    plt.legend(fontsize=22)

    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs(
            'results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/2mean'.format(args.arguments,
                                             args.num_repeats), dpi=300)


def plot_weight_decay_loss_acc(args):
    # Plot losses
    fig, ax = plt.subplots(figsize=(6.35, 4.5))
    for weight_decay, color in [(0.0005, 'C0'), (0.00009, 'C3'), (0, 'C1')]:
        losses = read_losses(args, weight_decay)
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
        x = np.arange(args.epochs)
        if weight_decay == 0:
            plt.plot(x[::2], val_loss.mean(axis=0)[::2], 'o-', color=color,
                     markersize=3, lw=1.5, label='$\lambda = {}$'.format(weight_decay))
        elif weight_decay == 0.00009:
            plt.plot(x[::2], val_loss.mean(axis=0)[::2], 'o-', color=color,
                     markersize=3, lw=1.5, label='$\lambda = {:.5f}$'.format(weight_decay))
        else:
            plt.plot(x[::2], val_loss.mean(axis=0)[::2], 'o-', color=color,
                     markersize=3, lw=1.5, label='$\lambda = {:.4f}$'.format(weight_decay))
    plt.ylim(0, 0.8)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Test loss', fontsize=24)
    plt.legend(fontsize=17.5, loc='lower right')

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig(
        'results/{}_{}/3loss'.format(args.arguments, args.num_repeats), dpi=300)

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for weight_decay, color in [(0.0005, 'C0'), (0.00009, 'C3'), (0, 'C1')]:
        losses = read_losses(args, weight_decay)
        val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                             for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])

        x = np.arange(args.epochs)
        if weight_decay == 0:
            plt.plot(x[::2], val_acc.mean(axis=0)[::2], 'o-', color=color,
                     markersize=3, lw=1.5, label='$\lambda = {}$'.format(weight_decay))
        elif weight_decay == 0.00009:
            plt.plot(x[::2], val_acc.mean(axis=0)[::2], 'o-', color=color,
                     markersize=3, lw=1.5, label='$\lambda = {:.5f}$'.format(weight_decay))
        else:
            plt.plot(x[::2], val_acc.mean(axis=0)[::2], 'o-', color=color,
                     markersize=3, lw=1.5, label='$\lambda = {:.4f}$'.format(weight_decay))
    plt.ylim(0.8, 0.92)
    # plt.ylim(0.84, 0.9)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    ax.set_yticks([0.82, 0.86, 0.90])
    # ax.set_yticks([0.84, 0.86, 0.88, 0.90])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Test accuracy', fontsize=24)
    plt.legend(fontsize=18)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs(
            'results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig(
        'results/{}_{}/3acc'.format(args.arguments, args.num_repeats), dpi=300)


def prepare_xy(args):
    '''
    '''
    num_bins = 100
    if args.activation_func == 'relu':
        # Prepare xy axis
        betas = [i * 2 / num_bins for i in range(1, num_bins+1)]
        J0s = [i * 5 / num_bins for i in range(-num_bins, 21)]
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i,
                                           beta in enumerate(betas[19:])]
        J0s = [str(J0) if i % 24 == 0 else '' for i, J0 in enumerate(J0s)]
        # Prepare data to plot
        fname = 'rawdata/100_100_100_relu'
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)
        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 100] = 100
        separation1_log = np.log10(np.abs(separation1)+0.0001)

    elif args.activation_func == 'tanh':
        # Prepare xy axis
        betas = [i * 3 / num_bins for i in range(1, num_bins+1)]
        J0s = [i * 2 / num_bins for i in range(-50, 51)]
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i,
                                           beta in enumerate(betas[19:])]
        J0s = [str(J0) if i % 20 == 0 else '' for i,
               J0 in enumerate(J0s)]
        # Prepare data to plot
        fname = 'rawdata/50_100_100_tanh'
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)
        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 1] = 1
        separation1_log = np.log10(np.abs(separation1)+0.0001)[:, 50:151]

    return betas, J0s, separation1_log


def read_weights(args, weight_decay):
    """Read the files contains weights and weights difference.

    Args:
        args: configuration options dictionary

    Returns:
        weights and weights difference
    """
    results = {}
    for num_repeat in range(args.num_repeats):
        results[num_repeat] = {}
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
            args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        for epoch in args.log_epochs:
            fname = 'rawdata/weights/{}/{}/epoch{:05d}'.format(arguments,
                                                               num_repeat, epoch)
            with open(fname, 'rb') as f:
                d = pickle.load(f)
            results[num_repeat][epoch] = d

    return results


def read_losses(args, weight_decay):
    """Get the losses from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    losses = {}
    for num_repeat in range(args.num_repeats):
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
            args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        fname = 'rawdata/losses/{}/{}/losses'.format(arguments, num_repeat)
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        losses[num_repeat] = d

    return losses


def read_scores(args, weight_decay):
    """Get the scores from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    scores = {}
    for num_repeat in range(args.num_repeats):
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
            args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        fname = 'rawdata/scores/{}/{}/scores'.format(arguments, num_repeat)
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        scores[num_repeat] = d

    return scores
