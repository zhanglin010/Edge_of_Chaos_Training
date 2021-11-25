import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import seaborn as sns
from matplotlib.lines import Line2D


def plot_boundary(args, fname):
    '''Plot the result
    '''
    # Prepare xy axis and separation data
    betas, J0s, integrals, mask = prepare_xy_boundary(args, fname)
    fig, ax = plt.subplots(1, figsize=(8, 6.5))
    # Plot order-chaos boundary using data.mat file
    ax = sns.heatmap(integrals, xticklabels=J0s,
                     yticklabels=betas[::-1], mask=mask, cmap='RdYlBu', cbar=False, vmin=0, vmax=1)
    # RdYlBu, GnBu
    ax.set_xticklabels(
        ax.get_xticklabels(), fontsize=24, rotation=0)
    ax.set_yticklabels(
        ax.get_yticklabels(), fontsize=24, rotation=0)
    ax.set_xlabel('$J_0/\ J$', fontsize=24)
    ax.set_ylabel('$1/\ J$', fontsize=24)

    fig.tight_layout()

    # Save the figure
    if not os.path.exists('results/separation/{}'.format(args.activation_func)):
        os.makedirs('results/separation/{}'.format(args.activation_func))
    plt.savefig(
        'results/separation/{}/{}_{}_boundary'.format(args.activation_func, args.dir, args.num_iter), dpi=300)


def plot_separation(args, fname):
    '''Plot the result
    '''
    # Prepare xy axis and separation data
    betas, J0s, separation1_log, integrals, mask = prepare_xy(args, fname)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Plot separation
    ax = sns.heatmap(separation1_log, xticklabels=J0s,
                     yticklabels=betas[::-1], cmap='Purples')
    ax.set_xticklabels(
        ax.get_xticklabels(), fontsize=24, rotation=0)
    ax.set_yticklabels(
        ax.get_yticklabels(), fontsize=24, rotation=0)
    # ax.set_title(
    #     'Sensitivity under perturbation $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', fontsize=24)

    # Plot colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(
        'Asymptotic distance $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', fontsize=24)
    if args.activation_func == 'relu':
        cbar_labels1 = [-5, -2, 0, 2]
        cbar.set_ticks(cbar_labels1)
        cbar_labels2 = ['< {}'.format(.00001), str(
            0.01), str(1.0), '>= {}'.format(100)]
        cbar.set_ticklabels(cbar_labels2)
        cbar.ax.tick_params(labelsize=22)
    elif args.activation_func == 'tanh':
        cbar_labels1 = [-4, -2, 0]
        cbar.set_ticks(cbar_labels1)
        cbar_labels2 = ['< {}'.format(.0001), str(0.01), '>= {}'.format(1)]
        cbar.set_ticklabels(cbar_labels2)
        cbar.ax.tick_params(labelsize=24)

    # Plot order-chaos boundary using data.mat file
    sns.heatmap(integrals, xticklabels=J0s,
                yticklabels=betas[::-1], mask=mask, cmap='bwr', cbar=False, vmin=0, vmax=1)
    ax.set_xlabel('$J_0/\ J$', fontsize=24)
    ax.set_ylabel('$1/\ J$', fontsize=24)
    # Manually add a legend
    handles, labels = ax.get_legend_handles_labels()
    lines = Line2D([0], [0], color='red', linewidth=5, linestyle='-')
    label = 'Edge of chaos'
    handles.insert(0, lines)
    labels.insert(0, label)
    plt.legend(handles=handles, labels=labels, fontsize=24, loc='upper left')

    fig.tight_layout()

    # Save the figure
    if not os.path.exists('results/separation/{}'.format(args.activation_func)):
        os.makedirs('results/separation/{}'.format(args.activation_func))
    plt.savefig(
        'results/separation/{}/{}_{}_{}'.format(args.activation_func, args.dir, args.num_iter, args.activation_func), dpi=300)


def prepare_xy(args, fname):

    if args.activation_func == 'relu':
        # Prepare xy axis
        betas = [i * 2 / args.num_bins for i in range(1, args.num_bins+1)]
        J0s = [i * 5 / args.num_bins for i in range(-args.num_bins, 21)]
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i,
                                           beta in enumerate(betas[19:])]
        J0s = [str(J0) if i % 24 == 0 else '' for i, J0 in enumerate(J0s)]
        # Prepare separation data to plot
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)

        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 100] = 100
        separation1_log = np.log10(np.abs(separation1)+0.00001)
        # Prepare boundary data to plot
        mat = spio.loadmat(
            'rawdata/{}/data_100bins_relu.mat'.format(args.activation_func), squeeze_me=True)
        integrals = mat['integrals'][::-1]
        mask = (integrals - 1) * (-1)

    elif args.activation_func == 'tanh':
        # Prepare boundary data to plot
        betas = [i * 2 / args.num_bins for i in range(1, args.num_bins+1)]
        J0s = [
            i * 2 / args.num_bins for i in range(-int(args.num_bins/2), args.num_bins+1)]
        betas = ['' for i in range(24)] + [str(beta) if i % 25 == 0 else '' for i,
                                           beta in enumerate(betas[24:])]
        J0s = [str(J0) if i % 50 == 0 else '' for i, J0 in enumerate(J0s)]
        # Prepare data to plot
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)

        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 1] = 1
        separation1_log = np.log10(np.abs(separation1)+0.0001)[:, :]
        # Prepare boundary data to plot
        mat = spio.loadmat(
            'rawdata/{}/data_100bins_tanh_modified.mat'.format(args.activation_func), squeeze_me=True)
        integrals = mat['integrals_modified'][::-1]
        integrals = integrals[:, :]
        mask = (integrals - 1) * (-1)

    return betas, J0s, separation1_log, integrals, mask


def prepare_xy_boundary(args, fname):
    if args.activation_func == 'relu':
        # Prepare xy axis
        betas = [i * 2 / args.num_bins for i in range(1, args.num_bins+1)]
        J0s = [i * 5 / args.num_bins for i in range(-args.num_bins, 21)]
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i,
                                           beta in enumerate(betas[19:])]
        J0s = [str(J0) if i % 24 == 0 else '' for i, J0 in enumerate(J0s)]
        # Prepare separation data to plot
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)

        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 100] = 100
        separation1_log = np.log10(np.abs(separation1)+0.00001)
        # Prepare boundary data to plot
        mat = spio.loadmat(
            'rawdata/{}/data_100bins_relu.mat'.format(args.activation_func), squeeze_me=True)
        integrals = mat['integrals'][::-1]
        mask = (integrals - 1) * (-1)

    elif args.activation_func == 'tanh':
        # Prepare boundary data to plot
        betas = [i * 2 / args.num_bins for i in range(1, args.num_bins+1)]
        J0s = [
            i * 2 / args.num_bins for i in range(-int(args.num_bins/2), args.num_bins+1)]
        betas = ['' for i in range(24)] + [str(beta) if i % 25 == 0 else '' for i,
                                           beta in enumerate(betas[24:])]
        J0s = [str(J0) if i % 50 == 0 else '' for i, J0 in enumerate(J0s)]
        # Prepare boundary data to plot
        mat = spio.loadmat(
            'rawdata/{}/data_100bins_tanh.mat'.format(args.activation_func), squeeze_me=True)
        integrals = mat['integrals'][::-1]
        integrals = integrals[:, :]
        mask = (integrals - 1) * (-1)

    return betas, J0s, integrals, mask
