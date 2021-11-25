import argparse
import time

from calculate_separation import calculate_separation
from plot import plot_separation, plot_boundary

parser = argparse.ArgumentParser(description='Phase_diagram')
parser.add_argument('--num-neurons', type=int, default=100, metavar='N',
                    help='number of neurons (default: 100)')
parser.add_argument('--activation-func', type=str, default='relu',
                    help='activation function for hidden layers')
parser.add_argument('--num-bins', type=int, default=5, metavar='N',
                    help='number of bins to plot the diagram (default: 40)')
parser.add_argument('--num-iter', type=int, default=50, metavar='N',
                    help='number of iterations (default: 50)')


def main():
    '''Main program
    '''
    start_time = time.time()
    args = parser.parse_args()

    args.dir = '{}_{}'.format(args.num_neurons, args.num_bins)
    fname = 'rawdata/{}/{}_{}'.format(args.activation_func,
                                      args.dir, args.num_iter)
    calculate_separation(args, args.num_iter, fname)
    # Plot the Fig.1(b) in the paper
    plot_separation(args, fname)
    # Plot the theoretical boundary only
    plot_boundary(args, fname)

    end_time = time.time()
    print('running time is {} mins'.format((end_time - start_time)/60))


if __name__ == '__main__':
    main()
