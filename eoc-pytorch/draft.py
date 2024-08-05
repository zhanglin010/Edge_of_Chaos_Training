'''plot_figure2.py'''
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.optimize import curve_fit

def read_weights(args, weight_decay):
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
    losses = []
    for num_repeat in range(args.num_repeats):
        fname = os.path.join(args.save_losses_dir, f'{num_repeat}', 'losses')
        with open(fname, 'rb') as f:
            losses.append(pickle.load(f))
    return losses

def plot_weights_path(args):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    results = read_weights(args, args.weight_decay)
    losses = read_losses(args, args.weight_decay)
    
    W0s = np.array([[results[num_repeat][epoch]['W0s'] 
                     for epoch in range(args.epochs//1)] 
                     for num_repeat in range(args.num_repeats)])
    val_loss = np.array([[losses[num_repeat]['val'][epoch][1] 
                          for epoch in range(args.epochs//1)] 
                          for num_repeat in range(args.num_repeats)])
    
    J = (np.std(W0s.mean(axis=0), axis=(1, 2)) * np.sqrt(args.input_shape))
    J0 = np.mean(W0s.mean(axis=0), axis=(1, 2)) * args.input_shape

    if args.activation_func == 'relu':
        x, y = (J0/J)*20+100.5, 100.5-(1/J)*50
        plt.plot(x, y, 'o-', color='C1', markersize=3.5, lw=2, label='Model evolution path')
        x_optimal = ((J0/J)*20+100.5)[np.argmin(val_loss)]
        y_optimal = (100.5-(1/J)*50)[np.argmin(val_loss)]
        plt.scatter(x_optimal, y_optimal, s=300, marker='o', color='C2', label='Optimal epoch', zorder=10)
        plt.scatter(x[-1], y[-1], s=300, marker='s', color='C1', label='Epoch {}'.format(args.epochs-1))
    elif args.activation_func == 'tanh':
        x, y = (J0/J)*50+50.5, 100.5-(1/J)*33.3333
        plt.plot(x, y, 'o-', color='C1', markersize=3, lw=2, label='Model evolution path')
        plt.scatter(x[-1], y[-1], s=300, marker='s', color='C1', label='Epoch {}'.format(args.epochs-1))
        x_optimal = ((J0/J)*50+50.5)[np.argmin(val_loss)]
        y_optimal = (100.5-(1/J)*33.3333)[np.argmin(val_loss)]
        plt.scatter(x_optimal, y_optimal, s=300, marker='o', color='C2', label='Optimal epoch (test loss)', zorder=10)
    
    # Plot arrow
    u, v = np.diff(x), np.diff(y)
    pos_x, pos_y = x[:-1] + u/2, y[:-1] + v/2
    norm = np.sqrt(u**2 + v**2)
    plt.quiver(pos_x[0], pos_y[0], (u/norm)[0], (v/norm)[0], width=0.002, lw=0.5, scale=20, headwidth=20, headlength=20, headaxislength=16, angles="xy", pivot="mid", color='C1')

    handles, labels = ax.get_legend_handles_labels()
    lines = Line2D([0], [0], color='black', linewidth=3, linestyle='--')
    label = 'Edge of chaos ( $J=1$ )'
    handles.insert(0, lines)
    labels.insert(0, label)
    plt.legend(handles=handles, labels=labels, fontsize=20)
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24)
    ax.set_ylabel(r'$1/\ J$', fontsize=24)

    fig.tight_layout()
    results_dir = f'results/{args.arguments}_{args.num_repeats}'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/path', dpi=300)

def plot_weights_var_mean(args):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    results = read_weights(args, args.weight_decay)
    losses = read_losses(args, args.weight_decay)
    
    W0s = np.array([[results[num_repeat][epoch]['W0s']
                     for epoch in range(args.epochs//1)] 
                     for num_repeat in range(args.num_repeats)])
    val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                          for epoch in range(args.epochs//1)] 
                          for num_repeat in range(args.num_repeats)])

    J = (np.std(np.array(W0s.mean(axis=0)), axis=(1, 2)) * np.sqrt(args.input_shape))
    J_squard = np.power(J, 2)
    x = np.arange(args.epochs)
    ax.plot(x[::2], J_squard[::2], 'o', markersize=3, color='C1')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))], J_squard[np.argmin(val_loss.mean(axis=0))], s=200, marker='o', color='C2', zorder=4, label='Optimal epoch')
    
    ax.set_xticks([0, 250, 500])
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(r'$J^2$', fontsize=24)
    plt.legend(fontsize=17, loc='upper left')

    axins = inset_axes(ax, width='58%', height='58%', loc='lower left', bbox_to_anchor=(.38, .065, 1, 1), bbox_transform=ax.transAxes)
    axins.axis([-4, 51, 0.2, 1.])
    axins.plot(x[:50:2], J_squard[:50:2], 'o', markersize=4.5, color='C1')
    plot_lin_regress(args, x[:50:2], J_squard[:50:2])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins.yaxis.get_major_locator().set_params(nbins=3)
    axins.xaxis.get_major_locator().set_params(nbins=3)
    axins.tick_params(axis='both', which='major', labelsize=13)

    fig.tight_layout()
    results_dir = f'results/{args.arguments}_{args.num_repeats}'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/variance', dpi=300)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    J0 = np.mean(np.array(W0s.mean(axis=0)), axis=(1, 2)) * args.input_shape
    ax.plot(J0, 'o', markersize=3)
    plt.scatter(np.argmin(val_loss.mean(axis=0)), J0[np.argmin(val_loss.mean(axis=0))], s=200, marker='o', color='C2', zorder=4, label='Optimal epoch')
    
    ax.set_xticks([0, 250, 500])
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(r'$J_0$', fontsize=24)
    plt.legend(fontsize=17, loc='lower right')
    
    fig.tight_layout()
    plt.savefig(f'{results_dir}/mean', dpi=300)
    plt.show()

def plot_lin_regress(args, x, y):
    def func(x, a, b, c):
        return a * np.log(b * x) + c

    popt, _ = curve_fit(func, x[1:], y[1:], p0=(0.1, 0.1, 0.1))
    plt.plot(x[1:], func(x[1:], *popt), '--', color='C2', lw=1.5)
    plt.scatter(x[1:], func(x[1:], *popt), s=10, color='C2')
    return popt

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Plot figures.')
    parser.add_argument('--arguments', type=str, default='default', help='Arguments used in the experiment.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay value.')
    parser.add_argument('--input_shape', type=int, default=32, help='Input shape size.')
    parser.add_argument('--save_weights_dir', type=str, required=True, help='Directory to save weights.')
    parser.add_argument('--save_losses_dir', type=str, required=True, help='Directory to save losses.')
    parser.add_argument('--activation_func', type=str, choices=['relu', 'tanh'], default='relu', help='Activation function used in the network.')
    
    args = parser.parse_args()
    
    with open(f'{args.save_weights_dir}/hyperparams.json', 'r') as f:
        hyperparams = json.load(f)
        args.input_shape = hyperparams.get('input_shape', args.input_shape)
        args.activation_func = hyperparams.get('activation_func', args.activation_func)
    
    plot_weights_path(args)
    plot_weights_var_mean(args)
