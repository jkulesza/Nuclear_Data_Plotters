#!/usr/bin/env python

# Customize plot style.
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['backend'] = 'Agg'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.figsize'] = (6.5, 4.017220927)
mpl.rcParams['axes.prop_cycle'] = cycler("color", ['#000000'])
mpl.rcParams['hatch.linewidth'] = 0.1
import matplotlib.pyplot as plt
import numpy as np

class xs_data:

    def __init__(self, infilename):
        self.infilename = infilename
        self.read_lsl_file()
        return

    def read_lsl_file(self):
        import re

        # Read in associated filename for processing.
        with open(self.infilename, 'r') as myfile:
            infile = myfile.read()

        # Get number of groups and group boundaries.
        tmp = re.search(r'number of energies plus 1.*?\n(.*?)\n+\*', infile, re.S).group(1)
        self.n_egrp_bnd = int(tmp)
        self.n_egrp = self.n_egrp_bnd - 1

        # Get energy grid.
        tmp = re.search(r'energy grid.*?\n(.*?)\n+\*', infile, re.S).group(1).split()
        self.l_egrid = np.array([float(i) for i in tmp])
        assert(len(self.l_egrid) == self.n_egrp_bnd)

        # Get cross section.
        tmp = re.search(r'cross section.*?\n(.*?)\n+\*', infile, re.S).group(1).split()
        self.l_xs = np.array([float(i) for i in tmp])
        assert(len(self.l_xs) == self.n_egrp)

        # Get standard deviation as percentage.
        tmp = re.search(r'standard deviation.*?\n(.*?)\n+\*', infile, re.S).group(1).split()
        self.l_stddev = np.array([float(i) for i in tmp])
        assert(len(self.l_stddev) == self.n_egrp)

        # Get and process correlation coefficients.
        tmp = re.search(r'correlation coefficient -- upper triangular.*?\n(.*?)$', infile, re.S).group(1).split()
        tmp = np.asarray([float(i) for i in tmp])
        self.l_cc = np.zeros((self.n_egrp, self.n_egrp))
        inds = np.triu_indices_from(self.l_cc)
        self.l_cc[inds] = tmp
        self.l_cc[(inds[1], inds[0])] = tmp

        return

def plot_correlation_coeffs(indata,
        emin=None,
        emax=None,
        nohatch=False,
        logscale_x = True,
        logscale_y = True,
        xlabel = 'Energy [eV]',
        ylabel = 'Energy [eV]',
        outfilename = 'tmp.png',
        cmap='bwr'):


    fig, (ax0) = plt.subplots(ncols=1)

    if(cmap=='parula'):
        from cmap_parula import cmap_parula
        cmap = cmap_parula().parula

    cax0 = ax0.pcolormesh(
        indata.l_egrid,
        indata.l_egrid,
        indata.l_cc,
        cmap=cmap,
        vmin=-1,
        vmax=1
    )

    if(not nohatch):
        indata.l_cc = np.ma.masked_where(indata.l_cc > 0, indata.l_cc)
        cax1 = ax0.pcolor(
            indata.l_egrid,
            indata.l_egrid,
            indata.l_cc,
            cmap=cmap,
            alpha=0,
            hatch='.......',
            vmin=0,
            vmax=1
        )

    # Set colorbar options.
    cbar0 = fig.colorbar(cax0, ax=ax0)

    if(emin == None):
        emin = np.min(indata.l_egrid)
    if(emax == None):
        emax = np.max(indata.l_egrid)

    ax0.set_aspect('equal')
    ax0.set_xlim([emin, emax])
    ax0.set_ylim([emin, emax])
    if(logscale_x):
        ax0.set_xscale('log')
    if(logscale_y):
        ax0.set_yscale('log')
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.savefig(outfilename, bbox_inches='tight')

    return

def get_y_limits(x, y, xmin, xmax, logscale_y = False):
    import warnings
    warnings.filterwarnings("ignore")

    # Get actual min and max values over displayed domain.
    ymin = y[np.logical_and(xmin < x, x < xmax)].min()
    ymax = y[np.logical_and(xmin < x, x < xmax)].max()

    # Get round number (on a log basis) ymin and ymax.
    if(logscale_y):
        ymin = 10**np.floor(np.log10(ymin))
        ymax = 10**np.ceil(np.log10(ymax))
    else:
        ymin = ymin - 0.1 * (ymax - ymin)
        ymax = ymax + 0.1 * (ymax - ymin)

    return ymin, ymax

def plot_fn(x, y,
    uncert_perc = None,
    outfilename = 'tmp.png',
    xmin = None,
    xmax = None,
    logscale_x = True,
    logscale_y = True,
    xlabel = 'Energy [eV]',
    ylabel = ''):

    import matplotlib.pyplot as plt
    fig, (ax0) = plt.subplots(ncols=1)
    y = np.hstack((y, y[-1]))

    if(uncert_perc != None):
        uncert_perc = uncert_perc * 0.01
        uncert_perc = np.hstack((uncert_perc, uncert_perc[-1]))
        uncert_abs = y * uncert_perc
        xu = np.ravel(np.array([[x[i], x[i+1]] for i in range(len(x)-1)]))
        yumin = np.ravel(np.array([[y[i+1]-uncert_abs[i+1], y[i+1]-uncert_abs[i+1]] for i in range(len(y)-1)]))
        yumax = np.ravel(np.array([[y[i+1]+uncert_abs[i+1], y[i+1]+uncert_abs[i+1]] for i in range(len(y)-1)]))
        ax0.fill_between(xu, yumin, yumax, color='#000000', alpha=0.2, lw=0)

    ax0.step(x, y)

    if(xmin == None):
        xmin = np.min(indata.l_egrid)

    if(xmax == None):
        xmax = np.max(indata.l_egrid)

    ax0.set_xlim([xmin, xmax])

    if(logscale_x):
        ax0.set_xscale('log')

    if(logscale_y):
        ax0.set_yscale('log')

    ymin, ymax = get_y_limits(x, y, xmin, xmax, logscale_y = logscale_y)
    ax0.set_ylim([ymin, ymax])
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.savefig(outfilename, bbox_inches='tight')

    return

import __main__ as main
if(__name__ == '__main__' and hasattr(main, '__file__')):

    ############################################################
    # Command line parsing.
    ############################################################

    import argparse
    import os
    import textwrap

    description = textwrap.dedent(
    """
    This script is used to read .lsl nuclear data files and to plot the data
    within.
    """)

    epilog = textwrap.dedent(
    """
    Typical command line calls might look like:

    > python """ + os.path.basename(__file__) + """ s32_IRDFF_103_nug89_tpl.lsl --emin 1e6
    """ + u"\u2063")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description, epilog=epilog)

    # Add required, positional, arguments.
    parser.add_argument(
        'inp', type = str,
        help = 'input file name (e.g., s32_IRDFF_103_nug89_tpl.lsl)')

    # Add optional, named, arguments.
    parser.add_argument(
        '--cmap', type = str, default = 'bwr',
        help = 'colormap for correlation coefficient plot (default: bwr)')
    parser.add_argument(
        '--nohatch', action = 'store_true',
        help = 'disable hatching on correlation coefficient plot (default: false)')
    parser.add_argument(
        '--emin', type = float, default = None,
        help = 'minimum energy for domain (default: min in file)')
    parser.add_argument(
        '--emax', type = float, default = None,
        help = 'maximum energy for domain (default: max in file)')

    args = parser.parse_args()

    # Basic error checking of command line options.
    assert(os.path.isfile(args.inp)), 'Input file (' + args.inp + ') not found.'

    ############################################################
    # Program execution.
    ############################################################

    indata = xs_data(args.inp)

    plot_correlation_coeffs(indata,
        emin=args.emin, emax=args.emax,
        cmap=args.cmap,
        nohatch=args.nohatch,
        outfilename = '{:}_cc.png'.format(indata.infilename))

    plot_fn(indata.l_egrid, indata.l_xs,
        xmin=args.emin, xmax=args.emax,
        outfilename = '{:}_xs.png'.format(indata.infilename),
        logscale_y = True,
        ylabel = 'Cross Section [b]')

    plot_fn(indata.l_egrid, indata.l_xs, uncert_perc = indata.l_stddev,
        xmin=args.emin, xmax=args.emax,
        outfilename = '{:}_xs_stddev.png'.format(indata.infilename),
        logscale_y = True,
        ylabel = 'Cross Section $\pm$ $1\sigma$ [b]')

    plot_fn(indata.l_egrid, indata.l_stddev,
        xmin=args.emin, xmax=args.emax,
        outfilename = '{:}_stddev.png'.format(indata.infilename),
        logscale_y = False,
        ylabel = 'Standard Deviation [%]')

