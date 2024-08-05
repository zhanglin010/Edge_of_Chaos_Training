import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.optimize import curve_fit

PATH = "D:/pycode/Edge_of_Chaos/eoc-pytorch/"


def plot_weights_path(args):
    '''Plot the path of this hyperparameter setting.'''

    # Plot model evolution path
    fig, ax = plt.subplots(figsize=(14, 7.5))
    betas, J0s, separation1_log = prepare_xy(args)
    
    # Plot separation
    ax = sns.heatmap(separation1_log, xticklabels=J0s, yticklabels=betas[::-1], 
                     cmap='Purples')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, rotation=0)
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24)
    ax.set_ylabel(r'$1/\ J$', fontsize=24)

    # Plot colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label( # type: ignore
        'Asymptotic distance $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', # type: ignore
        fontsize=24
        )

    if args.activation_func == 'relu':
        cbar_labels1 = [-4, -2, 0, 2]
        cbar_labels2 = ['< {}'.format(.0001), str(0.01), str(1.0), '>= {}'.format(100)]
    elif args.activation_func == 'tanh':
        cbar_labels1 = [-4, -2, 0]
        cbar_labels2 = ['< {}'.format(.0001), str(0.01), '>= {}'.format(1)]

    cbar.set_ticks(cbar_labels1) # type: ignore
    cbar.set_ticklabels(cbar_labels2) # type: ignore
    cbar.ax.tick_params(labelsize=24) # type: ignore

    # Plot theoretical boundary
    if args.activation_func == 'relu':
        mat = spio.loadmat(PATH + 'rawdata/data_100bins_relu.mat', squeeze_me=True)
        integrals = mat['integrals'][::-1]
        for i in range(0, integrals.shape[1], 2):
            integrals[:, i] = 0
        mask = (integrals - 1) * (-1)
        sns.heatmap(integrals, xticklabels=J0s, yticklabels=betas[::-1], mask=mask, 
                    cmap='gray', cbar=False)
    elif args.activation_func == 'tanh':
        plt.axhline(y=(100.5-1*33.3333), color='black', linestyle='--', lw=3)

    # Plot paths
    results = read_weights(args, args.weight_decay)
    # print(f'results shape: {np.array(results).shape}') # (1, 501)
    losses = read_losses(args, args.weight_decay)
    W0s = np.array([[results[num_repeat][epoch]['W0s'] 
                     for epoch in range(args.epochs//1)] 
                     for num_repeat in range(args.num_repeats)])
    # print(f'W0s shape: {W0s.shape}') # (1, 501, 784, 784)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0] 
                          for epoch in range(args.epochs//1)] 
                          for num_repeat in range(args.num_repeats)])
    # val_acc = np.array([[losses[num_repeat]['val'][epoch][2] 
    #                      for epoch in range(args.epochs//1)] 
    #                      for num_repeat in range(args.num_repeats)])

    J = (np.std(W0s.mean(axis=0), axis=(1, 2))*np.sqrt(args.input_shape))
    # print(f'J shape: {J.shape}') # (501,)
    J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape
    # print(f'J0 shape: {J0.shape}') # (501,)

    if args.activation_func == 'relu':
        x, y = (J0/J)*20+100.5, 100.5-(1/J)*50
        plt.plot(x, y, 'o-', color='C1',
                 markersize=3.5, lw=2, label='Model evolution path')
        x_optimal = ((J0/J)*20+100.5)[np.argmin(val_loss)]
        y_optimal = (100.5-(1/J)*50)[np.argmin(val_loss)]
        plt.scatter(x_optimal, y_optimal, s=300, marker='o', color='C2',
                    label='Optimal epoch', zorder=10)
        # plt.scatter(x[-1], y[-1], s=300, marker='s',
        #             color='C1', label='Epoch {}'.format(args.epochs-1))
    elif args.activation_func == 'tanh':
        x, y = (J0/J)*50+50.5, 100.5-(1/J)*33.3333
        plt.plot(x, y, 'o-', color='C1',
                 markersize=3, lw=2, label='Model evolution path')
        plt.scatter(x[-1], y[-1], s=300, marker='s',
                    color='C1', label='Epoch {}'.format(args.epochs-1))
        x_optimal = ((J0/J)*50+50.5)[np.argmin(val_loss.mean(axis=0))]
        y_optimal = (100.5-(1/J)*33.3333)[np.argmin(val_loss.mean(axis=0))]
        plt.scatter(x_optimal, y_optimal, s=300, marker='o', color='C2',
                    label='Optimal epoch (test loss)', zorder=10)

    # Plot arrow
    u, v = np.diff(x), np.diff(y)
    pos_x, pos_y = x[:-1] + u/2, y[:-1] + v/2
    norm = np.sqrt(u**2+v**2)
    plt.quiver(pos_x[0], pos_y[0], (u/norm)[0], (v/norm)[0], width=0.002, 
               lw=0.5, scale=20, headwidth=20, headlength=20, headaxislength=16, 
               angles="xy", pivot="mid", color='C1')
    
    # Manually add a legend
    handles, labels = ax.get_legend_handles_labels()
    lines = Line2D([0], [0], color='black', linewidth=3, linestyle='--')
    label = 'Edge of chaos ( $J=1$ )'
    handles.insert(0, lines)
    labels.insert(0, label)
    plt.legend(handles=handles, labels=labels, fontsize=20)
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24)
    ax.set_ylabel(r'$1/\ J$', fontsize=24)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/path'.format(args.arguments, args.num_repeats), dpi=300)


def plot_loss_acc(args):
    '''Plot loss and accuracy of this hyperparameter setting.'''

    # Plot losses
    fig, ax = plt.subplots(figsize=(6.25, 4.5))
    losses = read_losses(args, args.weight_decay)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] 
                          for num_repeat in range(args.num_repeats)])
    train_loss = np.array([[losses[num_repeat]['train'][epoch][0]
                            for epoch in range(args.epochs//1)] 
                            for num_repeat in range(args.num_repeats)])
    
    x = np.arange(args.epochs)
    plt.plot(x[::2], train_loss.mean(axis=0)[::2], 'o-', markersize=4, lw=1.5, 
             label='Train')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))],
                val_loss.mean(axis=0)[np.argmin(val_loss.mean(axis=0))],
                s=300, marker='o', color='C2', zorder=5, 
                label='Optimal epoch')  
    plt.plot(x[::2], val_loss.mean(axis=0)[::2], 'o-', markersize=4, lw=1.5, 
             label='Test')
    plt.ylim(-0.02, 1)
    ax.set_xticks([0, 250, 500])
    ax.set_yticks([0, 0.4, 0.8])

    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.legend(fontsize=19, loc='lower left', bbox_to_anchor=(0.35, 0.05))

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/loss'.format(args.arguments, args.num_repeats), dpi=300)

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(6.25, 4.5))
    val_acc = np.array([[losses[num_repeat]['val'][epoch][1]
                         for epoch in range(args.epochs//1)] 
                         for num_repeat in range(args.num_repeats)])
    train_acc = np.array([[losses[num_repeat]['train'][epoch][1]
                           for epoch in range(args.epochs//1)] 
                           for num_repeat in range(args.num_repeats)])

    x = np.arange(args.epochs)
    plt.plot(x[::2], train_acc.mean(axis=0)[::2], 'o-', markersize=4, lw=1.5, 
             label='Train')
    plt.plot(x[::2], val_acc.mean(axis=0)[::2], 'o-', markersize=4, lw=1.5, 
             label='Test')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))], 
                val_acc.mean(axis=0)[np.argmin(val_loss.mean(axis=0))],
                s=300, marker='o', color='C2', zorder=5, label='Optimal epoch')
    plt.ylim(0.7, 1.01)
    ax.set_xticks([0, 250, 500])

    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.legend(fontsize=20)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/acc'.format(args.arguments, args.num_repeats), dpi=300)


def plot_weights_var_mean(args):
    '''Plot variance and mean of the weights of this hyperparameter setting.'''

    # plot variance
    fig, ax = plt.subplots(figsize=(6.5, 5))
    results = read_weights(args, args.weight_decay)
    losses = read_losses(args, args.weight_decay)
    W0s = np.array([[results[num_repeat][epoch]['W0s']
                     for epoch in range(args.epochs//1)] 
                     for num_repeat in range(args.num_repeats)])
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] 
                          for num_repeat in range(args.num_repeats)])
    # val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
    #                      for epoch in range(args.epochs//1)] 
    #                      for num_repeat in range(args.num_repeats)])

    J = (np.std(np.array(W0s.mean(axis=0)), axis=(1, 2))*np.sqrt(args.input_shape))
    J_squard = np.power(J, 2)
    x = np.arange(args.epochs)
    ax.plot(x[::2], J_squard[::2], 'o', markersize=3, color='C1')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))], 
                J_squard[np.argmin(val_loss.mean(axis=0))],
                s=200, marker='o', color='C2', zorder=4, label='Optimal epoch')
    # plt.scatter(x[np.argmax(val_acc.mean(axis=0))], 
    #             J_squard[np.argmax(val_acc.mean(axis=0))],
    #             s=200, marker='o', color='C3', zorder=5, label='Optimal epoch')
    # plot_lin_regress(args, J_squard)
    ax.set_xticks([0, 250, 500])
    ax.tick_params(axis='both', which='major', labelsize=24)
    # plt.ylim(0.2, 1.02)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(r'$J^2$', fontsize=24)
    plt.legend(fontsize=17, loc='upper left')

    axins = inset_axes(ax, width='58%', height='58%', loc='lower left',
                       bbox_to_anchor=(.38, .065, 1, 1), bbox_transform=ax.transAxes)
    # axins.axis([-4, 51, 0.2, 1.])
    # axins.plot(x[:50:2], J_squard[:50:2], 'o', markersize=4.5)
    # plot_lin_regress(args, x[:50:2], J_squard[:50:2])
    axins.axis([-4, 51, 0.2, 1.])
    axins.plot(x[:50:2], J_squard[:50:2], 'o', markersize=4.5, color='C1')
    plot_lin_regress(args, x[:50:2], J_squard[:50:2])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins.yaxis.get_major_locator().set_params(nbins=3)
    axins.xaxis.get_major_locator().set_params(nbins=3)
    # plt.setp(axins.get_xticklabels(), visible=False)
    # plt.setp(axins.get_yticklabels(), visible=False)
    # plt.legend(fontsize=13, loc='lower right')
    axins.tick_params(axis='both', which='major', labelsize=13)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/variance'.format(args.arguments, args.num_repeats), dpi=300)

    # plot mean
    fig, ax = plt.subplots(figsize=(6.5, 5))
    J0 = np.mean(np.array(W0s.mean(axis=0)), axis=(1, 2))*args.input_shape
    ax.plot(J0, 'o', markersize=3)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(r'$J_0$', fontsize=24)
    plt.ylim(top=1)
    plt.legend(fontsize=22)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/mean'.format(args.arguments, args.num_repeats), dpi=300)


def plot_lin_regress(args, x, y):
    '''Plot linear regression line.'''

    def func(x, a, b):
        return a * x + b
    
    # x_data = x
    # y_data = y
    # for end_epoch in [200]:
    #     x = x_data[:end_epoch]
    #     y = y_data[:end_epoch]

        # popt, pcov = curve_fit(func_0, x, y)
        # perr = np.sqrt(np.diag(pcov))
        # plt.plot(x, func_0(x, *popt), '-', linewidth=1.25, color='C3',
        #          label='Fit: ' + r'$A\times$' + 'Epoch + ' + r'$C$')

    popt, pcov = curve_fit(func, x, y)
    # perr = np.sqrt(np.diag(pcov))
    plt.plot(x, func(x, *popt), '-', linewidth=1.25, color='C3',
                label='Slope: {:.7f}'.format(popt[0]))


def prepare_xy(args):
    '''Prepare xy axis and data to plot.'''
    num_bins = 100
    if args.activation_func == 'relu':
        # Prepare xy axis
        betas = [i * 2 / num_bins for i in range(1, num_bins+1)]
        J0s = [i * 5 / num_bins for i in range(-num_bins, 21)]
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i,
                                           beta in enumerate(betas[19:])]
        J0s = [str(J0) if i % 24 == 0 else '' for i, J0 in enumerate(J0s)]
        # Prepare data to plot
        fname = PATH + 'rawdata/100_100_100_relu'
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
        fname = PATH + 'rawdata/50_100_100_tanh'
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)
        separation1 = results1[::-1, :, 0]
        separation1[separation1 > 1] = 1
        separation1_log = np.log10(np.abs(separation1)+0.0001)[:, 50:151]

    return betas, J0s, separation1_log


def read_weights(args, weight_decay):
    del weight_decay
    weights = []
    for num_repeat in range(args.num_repeats):
        repeat_weights = []
        for epoch in range(args.epochs):
            fname = os.path.join(args.save_weights_dir, f'{num_repeat}', f'epoch{epoch:05d}')
            with open(fname, 'rb') as f:
                repeat_weights.append(pickle.load(f))
        weights.append(repeat_weights)
    return weights


def read_losses(args, weight_decay):
    del weight_decay
    losses = []
    for _ in range(args.num_repeats):
        fname = os.path.join(args.save_losses_dir, 'losses')
        with open(fname, 'rb') as f:
            losses.append(pickle.load(f))
    return losses
