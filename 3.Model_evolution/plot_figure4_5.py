import os
import pickle

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset,
                                                   zoomed_inset_axes)
from scipy import stats
from scipy.optimize import curve_fit


def plot_rescale_same(args):
    '''Plot Fig.4 in the paper.
    '''
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for epochs, lr, batch_size, momentum, d in [(501, 0.005, 8, 0.5, 0.25), (501, 0.01, 16, 0.5, 0.5), (501, 0.02, 32, 0.5, 1), (501, 0.04, 64, 0.5, 2), (501, 0.08, 128, 0.5, 4)]:
        results = read_weights(args, epochs, lr, batch_size,
                               momentum, args.weight_decay)
        losses = read_losses(args, epochs, lr, batch_size,
                             momentum, args.weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(args.epochs)] for num_repeat in range(1)])
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(args.epochs)] for num_repeat in range(1)])

        J = (np.std(np.array(W0s.mean(axis=0)), axis=(1, 2))*28)
        J_squard = np.power(J, 2)
        ax.plot(np.arange(args.epochs)[::2], J_squard[::2],
                'o-', markersize=3, label='$d$'+'={}'.format(d))
        # plt.scatter(np.arange(args.epochs)[np.argmin(val_loss)], J_squard[np.argmin(val_loss)],
        #             s=100, marker='o', zorder=5)  # , label='Optimal epoch: c={}'.format(lr/(0.01*constant)))
        # plot_lin_regress(args, J_squard)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('$J^2$', fontsize=24)
    plt.legend(fontsize=24)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/combined_same_variance'.format(args.arguments,
                                                              args.num_repeats), dpi=300)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for epochs, lr, batch_size, momentum, d in [(501, 0.005, 8, 0.5, 0.25), (501, 0.01, 16, 0.5, 0.5), (501, 0.02, 32, 0.5, 1), (501, 0.04, 64, 0.5, 2), (501, 0.08, 128, 0.5, 4), ]:
        losses = read_losses(args, epochs, lr, batch_size,
                             momentum, args.weight_decay)
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(args.epochs)] for num_repeat in range(1)])

        plt.plot(np.arange(args.epochs)[::2], val_loss.mean(axis=0)[::2], 'o-',
                 markersize=3, lw=1.5, label='$d$'+'={}'.format(d))
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks([0, 250, 500])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Test loss', fontsize=24)
    plt.ylim(0, 0.8)
    # plt.legend(loc='upper left', fontsize=24)
    plt.legend(fontsize=24)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/combined_same_loss'.format(args.arguments,
                                                          args.num_repeats), dpi=300)


def plot_rescale_different(args):
    '''Plot Fig.5 in the paper.
    '''
    # variance of weight matrix
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for epochs, lr, batch_size, momentum, c2 in [(8001, 0.0025, 32, 0, 0.25), (4001, 0.005, 32, 0, 0.5), (2001, 0.01, 32, 0, 1), (1001, 0.01, 32, 0.5, 2), (501, 0.02, 32, 0.5, 4), ]:
        results = read_weights(args, epochs, lr, batch_size,
                               momentum, args.weight_decay)
        losses = read_losses(args, epochs, lr, batch_size,
                             momentum, args.weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(epochs)] for num_repeat in range(1)])
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(epochs)] for num_repeat in range(1)])
        val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                             for epoch in range(epochs)] for num_repeat in range(1)])

        J = (np.std(np.array(W0s.mean(axis=0)), axis=(1, 2))*28)
        J0 = np.mean(np.array(W0s.mean(axis=0)), axis=(1, 2))*784
        J_squard = np.power(J, 2)
        x = np.arange(epochs)*lr / ((1-momentum) * batch_size)
        ax.plot(x[::2], J_squard[::2],
                'o-', markersize=3, label='$c=$'+'{}'.format(c2))
        # plt.scatter(x[np.argmin(val_loss.mean(axis=0))], J_squard[np.argmin(val_loss.mean(axis=0))],
        #             s=100, marker='o', zorder=5)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_yticks([1, 2, 3, 4])
    plt.xlabel('Epoch' + r'$\times \frac {\eta}{(1-\alpha)B}$', fontsize=24)
    plt.ylabel('$J^2$', fontsize=24)
    plt.legend(fontsize=12, bbox_to_anchor=(-0.001, 0.54), loc='lower left')

    axins = inset_axes(ax, width='48%', height='48%', loc='lower left',
                       bbox_to_anchor=(.45, .045, 1, 1), bbox_transform=ax.transAxes)
    axins.axis([-0.005, 0.07, 0.2, 1.])
    for epochs, lr, batch_size, momentum, c2 in [(8001, 0.0025, 32, 0, 0.25), (4001, 0.005, 32, 0, 0.5), (2001, 0.01, 32, 0, 1), (1001, 0.01, 32, 0.5, 2), (501, 0.02, 32, 0.5, 4), ]:
        results = read_weights(args, epochs, lr, batch_size,
                               momentum, args.weight_decay)
        losses = read_losses(args, epochs, lr, batch_size,
                             momentum, args.weight_decay)
        W0s = np.array([[results[num_repeat][epoch]['W0s']
                         for epoch in range(epochs)] for num_repeat in range(1)])
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(epochs)] for num_repeat in range(1)])
        val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                             for epoch in range(epochs)] for num_repeat in range(1)])

        J = (np.std(np.array(W0s.mean(axis=0)), axis=(1, 2))*28)
        J0 = np.mean(np.array(W0s.mean(axis=0)), axis=(1, 2))*784
        J_squard = np.power(J, 2)
        x = np.arange(epochs)*lr / ((1-momentum) * batch_size)
        optimal_epoch = int((epochs-1)/10)
        print('optimal epoch is {}'.format(optimal_epoch))
        axins.plot(x[:optimal_epoch:2],
                   J_squard[:optimal_epoch:2], 'o-', markersize=3)
        # plt.scatter(x[np.argmin(val_loss.mean(axis=0))], J_squard[np.argmin(val_loss.mean(axis=0))],
        #             s=100, marker='o', zorder=5)
        # plot_lin_regress(args, x[:50:2], J_squard[:50:2])

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins.yaxis.get_major_locator().set_params(nbins=3)
    axins.xaxis.get_major_locator().set_params(nbins=3)
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/combined_variance_rescale'.format(args.arguments,
                                                                 args.num_repeats), dpi=300)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for epochs, lr, batch_size, momentum, c2 in [(8001, 0.0025, 32, 0, 0.25), (4001, 0.005, 32, 0, 0.5), (2001, 0.01, 32, 0, 1), (1001, 0.01, 32, 0.5, 2), (501, 0.02, 32, 0.5, 4), ]:
        losses = read_losses(args, epochs, lr, batch_size,
                             momentum, args.weight_decay)
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(epochs)] for num_repeat in range(1)])
        val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                             for epoch in range(epochs)] for num_repeat in range(1)])

        plt.plot(np.arange(epochs)[::2]*lr / ((1-momentum) * batch_size), val_loss.mean(axis=0)[::2], 'o-',
                 markersize=3, lw=1.5, label='$c=$'+'{}'.format(c2))  # label='Test loss, lr: {}, best loss: {:.04f}'.format(lr, test_loss[np.argmin(test_loss)]))
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    plt.xlabel('Epoch' + r'$\times \frac {\eta}{(1-\alpha)B}$', fontsize=24)
    plt.ylabel('Test loss', fontsize=22)
    plt.ylim(0.12, 0.82)
    plt.legend(fontsize=20)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/combined_loss_rescale'.format(args.arguments,
                                                             args.num_repeats), dpi=300)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for epochs, lr, batch_size, momentum, c2 in [(8001, 0.0025, 32, 0, 0.25), (4001, 0.005, 32, 0, 0.5), (2001, 0.01, 32, 0, 1), (1001, 0.01, 32, 0.5, 2), (501, 0.02, 32, 0.5, 4), ]:
        losses = read_losses(args, epochs, lr, batch_size,
                             momentum, args.weight_decay)
        val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                              for epoch in range(epochs)] for num_repeat in range(1)])
        val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                             for epoch in range(epochs)] for num_repeat in range(1)])
        plt.plot(np.arange(epochs)[::2]*lr / ((1-momentum) * batch_size), val_acc.mean(axis=0)[::2], 'o-',
                 markersize=3, lw=1.5, label='$c=$'+'{}'.format(c2))
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch' + r'$\times \frac {\eta}{(1-\alpha)B}$', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.ylim(0.8, 1)
    plt.legend(fontsize=20)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/combined_acc_rescale'.format(args.arguments,
                                                            args.num_repeats), dpi=300)


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
        fname = 'rawdata/100_100_500_tanh'
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)
        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 1] = 1
        separation1_log = np.log10(np.abs(separation1)+0.0001)[:, 50:151]

    return betas, J0s, separation1_log


def read_weights(args, epochs, lr, batch_size, momentum, weight_decay):
    """Read the files contains weights and weights difference.

    Args:
        args: configuration options dictionary

    Returns:
        weights and weights difference
    """
    results = {}
    log_epochs = np.arange(epochs)
    for num_repeat in range(args.num_repeats):
        results[num_repeat] = {}
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, epochs, int(lr*10000), int(
            batch_size), int(momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        for epoch in log_epochs:
            fname = 'rawdata/weights/{}/{}/epoch{:05d}'.format(arguments,
                                                               num_repeat, epoch)
            with open(fname, 'rb') as f:
                d = pickle.load(f)
            results[num_repeat][epoch] = d

    return results


def read_losses(args, epochs, lr, batch_size, momentum, weight_decay):
    """Get the losses from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    losses = {}
    for num_repeat in range(args.num_repeats):
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, epochs, int(lr*10000), int(
            batch_size), int(momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        fname = 'rawdata/losses/{}/{}/losses'.format(arguments, num_repeat)
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        losses[num_repeat] = d

    return losses


def read_scores(args, epochs, lr, batch_size, momentum, weight_decay):
    """Get the scores from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    scores = {}
    for num_repeat in range(args.num_repeats):
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, epochs, int(lr*10000), int(
            batch_size), int(momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        fname = 'rawdata/scores/{}/{}/scores'.format(arguments, num_repeat)
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        scores[num_repeat] = d

    return scores
