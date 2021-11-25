import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit


def main():
    # lr1 all linear scale
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)
    # Sample data
    # x = np.array([0.0005, 0.001, 0.002, 0.003, 0.005, 0.007,
    #               0.01, 0.02, 0.05, 0.07, 0.1])
    # y = np.array([0.000170, 0.000338, 0.000681, 0.00103, 0.00172, 0.00242,
    #               0.00342, 0.00688, 0.0171, 0.0238, 0.0345])
    x = np.array([0.0005, 0.001, 0.002, 0.003, 0.005, 0.007,
                  0.01, 0.02, 0.05, 0.1])
    y = np.array([0.000170, 0.000338, 0.000681, 0.00103, 0.00172, 0.00242,
                  0.00342, 0.00688, 0.0171, 0.0345])

    plt.plot(x, y, 'o', ms=8)
    plot_lin_regress(x, y)
    # plt.xscale('symlog', linthreshx=0.005)
    # plt.yscale('symlog', linthreshy=0.00156)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    # plt.minorticks_off()
    # ax.set_xticks([pow(10, -3), pow(10, -2), pow(10, -1)])
    # ax.set_yticks([pow(10, -4), pow(10, -3), pow(10, -2)])
    ax.set_xticks([pow(2, -10), pow(2, -7), pow(2, -4)])
    ax.set_yticks([pow(2, -11), pow(2, -8), pow(2, -5)])
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # plt.yticks([0, 0.02, 0.04])
    plt.ylabel(r'Slope $A$', fontsize='25')
    plt.xlabel(r'$\eta$', fontsize='25')
    plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.title(r'Coeeficient $A$ and momentum${\alpha}$', fontsize='22')
    # plt.legend(fontsize=22)
    fig.tight_layout()
    plt.savefig('results/coefficient/lr1', dpi=300)

    # momentum 1
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)
    # Sample data
    x = 1/(1-np.array([0, 0.3, 0.5, 0.6,
                       0.7, 0.8, 0.85, 0.9, 0.93, 0.95]))
    # print('alpha is {}'.format(x))
    y = np.array([0.00342, 0.00493, 0.00701, 0.00863, 0.0117, 0.0173, 0.0231,
                  0.0350, 0.0523, 0.0765])

    plt.plot(x, y, 'o', ms=8,)
    plot_lin_regress(x, y)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    ax.set_xticks([1, 4, 16])
    ax.set_yticks([pow(2, -8), pow(2, -6), pow(2, -4)])
    # plt.ylim([0.001, 0.08])
    # plt.locator_params(axis='x', nbins=5)
    # plt.locator_params(axis='y', nbins=4)
    # plt.yticks([0, 0.04, 0.08])
    plt.ylabel(r'Slope $A$', fontsize='25')
    plt.xlabel(r'$1/(1-\alpha)$', fontsize='25')
    plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.title(r'Coeeficient $A$ and momentum${\alpha}$', fontsize='22')
    fig.tight_layout()
    plt.savefig('results/coefficient/momentum1', dpi=300)

    # batch size
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111)
    x = 1/np.array([4, 8, 16, 24, 32, 64, 128, 192, 256, 512])
    y = np.array([0.0280, 0.0138, 0.00693, 0.00461, 0.00342, 0.00172,
                  0.000856, 0.000573, 0.000425, 0.000212])

    plt.plot(x, y, 'o', ms=8)
    plot_lin_regress(x, y)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    ax.set_xticks([pow(2, -8), pow(2, -5), pow(2, -2)])
    ax.set_yticks([pow(2, -11), pow(2, -8), pow(2, -5)])
    # plt.locator_params(axis='x', nbins=5)
    # plt.locator_params(axis='y', nbins=4)
    # plt.yticks([0, 0.004, 0.008])
    plt.ylabel(r'Slope $A$', fontsize='25')
    plt.xlabel(r'$1/B$', fontsize='25')
    plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.title(r'Coeeficient $A$ and momentum${\alpha}$', fontsize='22')
    fig.tight_layout()
    plt.savefig('results/coefficient/batchsize', dpi=300)


def plot_lin_regress(x, y):
    popt, pcov = curve_fit(linear_func, x, y)
    perr = np.sqrt(np.diag(pcov))
    # plt.plot(x, func(x, *popt), '-', color='C3',
    #          label='fit: ${:0.3f} * x^{1/2}+{:0.3f}$\nperr: {:0.3f}, {:0.3f}'.format(*popt, *perr))
    plt.plot(x, linear_func(x, *popt), '-', color='C3', lw=1.5,)
    # plt.legend()


def linear_func(x, a):
    return a * x


def plot_quadratic(x, y):
    popt, pcov = curve_fit(quadratic_func, x, y)
    perr = np.sqrt(np.diag(pcov))
    # plt.plot(x, func(x, *popt), '-', color='C3',
    #          label='fit: ${:0.3f} * x^{1/2}+{:0.3f}$\nperr: {:0.3f}, {:0.3f}'.format(*popt, *perr))
    plt.plot(x, quadratic_func(x, *popt), '-', color='C3',)
    # label='fit: ${:0.3f} * x^2'.format(*popt))
    plt.legend()


def quadratic_func(x, a):
    return a * np.square(x)


if __name__ == "__main__":
    main()
