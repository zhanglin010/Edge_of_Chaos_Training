import os
import pickle

import numpy as np


def calculate_separation(args, num_iter, fname):
    '''

    '''
    J = 1.0
    results = []
    for i in range(1, args.num_bins+1):
        result = []
        _beta = J * i * 2.0 / args.num_bins
        if args.activation_func == 'relu':
            for j in range(-args.num_bins, 21):
                _J0 = J * j * 5.0 / args.num_bins
                W = set_weight(L=args.num_neurons, J=J, J0=_J0, beta=_beta)
                array = get_chaos(args, W, num_iter)
                result.append(array)
        elif args.activation_func == 'tanh':
            for j in range(-int(args.num_bins/2), args.num_bins+1):
                _J0 = J * j * 2.0 / args.num_bins
                W = set_weight(L=args.num_neurons, J=J, J0=_J0, beta=_beta)
                array = get_chaos(args, W, num_iter)
                result.append(array)
        results.append(result)

    results = np.array(results)

    if not os.path.exists('rawdata/{}'.format(args.activation_func)):
        os.makedirs('rawdata/{}'.format(args.activation_func))
    with open(fname, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def set_weight(L, J, J0, beta):
    N = L * L
    return np.random.normal(J0/N, J/L, (N, N)) / beta


def get_chaos(args, W, num_iter):
    '''

    Args:
        W: weight matrix
        num_iter: number of iterations

    Return:
        A numpy array whose
        1st and 2nd are the final num_iter/10 time steps' average divergences,
        2 are used to check convergence.
        3rd and 4th elements are the mean value of the final time steps separated
        by num_iter/10 used to get the P/F/SG states.
    '''
    N = W.shape[0]
    # x0
    x0 = np.random.normal(0, 1, N)
    x = x0
    xs = []
    for i in range(num_iter):
        x = np.matmul(x, W)
        if args.activation_func == 'relu':
            x = np.maximum(x, 0)
        elif args.activation_func == 'tanh':
            x = np.tanh(x)
        xs.append(x)

    # x1
    x1 = x0 + np.random.normal(0, 0.0001, N)
    # x1 = x0 + 0.0001 * np.ones(N)
    x = x1
    xs1 = []
    for i in range(num_iter):
        x1 = np.matmul(x1, W)
        if args.activation_func == 'relu':
            x1 = np.maximum(x1, 0)
        elif args.activation_func == 'tanh':
            x1 = np.tanh(x1)
        xs1.append(x1)

    # separation
    separation = np.mean(np.linalg.norm(np.array(xs) -
                                        np.array(xs1), axis=1)[-num_iter//5:])
    u = np.mean(xs[-num_iter//5:])
    b = np.mean(xs[-num_iter//5:], axis=0)
    q0 = np.mean(np.power(b, 2))

    return np.array([separation, u, q0])
