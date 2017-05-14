#!/usr/bin/env python

# Customize plot style.
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['backend'] = 'Agg'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
# mpl.rcParams['axes.color_cycle'] = ['#000000', 'grey']
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.figsize'] = (6.5, 4.017220927)
mpl.rcParams['axes.prop_cycle'] = cycler("color", ['#000000'])
import matplotlib.pyplot as plt
import numpy as np

class xs_data_eval:

    def __init__(self, infilename):
        self.infilename = infilename
        self.read_zvd_eval_file()
        return

    def read_zvd_eval_file(self):
        import re

        # Read in associated filename for processing.
        with open(self.infilename, 'r') as myfile:
            infile = myfile.read()

        # Get header data.
        self.daytime = re.search(r'ZVView-data-copy:\s+(.*?)\n', infile, re.S).group(1)
        self.name = re.search(r'name:\s+(.*?)\n', infile, re.S).group(1)
        self.Xaxis = re.search(r'X.axis:\s+(.*?)\n', infile, re.S).group(1)
        self.Yaxis = re.search(r'Y.axis:\s+(.*?)\n', infile, re.S).group(1)
        self.wdata = int(re.search(r'wdata:\s+(.*?)\n', infile, re.S).group(1))
        self.ldata = int(re.search(r'ldata:\s+(.*?)\n', infile, re.S).group(1))

        # Get field and field units.
        tmp = re.search(r'data\.\.\..*?\n(.*?)\n(.*?)\n', infile, re.S)
        self.fields = tmp.group(1)
        self.field_units = tmp.group(2)

        self.fields = self.fields.split()[1:]
        self.field_units = self.field_units.split()[1:]

        self.xlabel = self.Xaxis + ' [' + self.field_units[0] + ']'
        self.ylabel = self.Yaxis + ' [' + self.field_units[1] + ']'

        # Get data.
        tmp = re.sub(r'#.*?\n', '', infile, re.S)
        tmp = re.sub(r'\n\/\/\n', '', tmp, re.S)
        tmp = tmp.split()
        self.X = tmp[0::2]
        self.Y = tmp[1::2]

        return

class xs_data_exp:

    def __init__(self, infilename):
        self.infilename = infilename
        self.read_zvd_exp_file()
        return

    def read_zvd_exp_file(self):
        import re

        # Read in associated filename for processing.
        with open(self.infilename, 'r') as myfile:
            infile = myfile.read()

        # Get header data.
        self.daytime = re.search(r'ZVView-data-copy:\s+(.*?)\n', infile, re.S).group(1)
        self.name = re.search(r'name:\s+(.*?)\n', infile, re.S).group(1)
        self.Xaxis = re.search(r'X.axis:\s+(.*?)\n', infile, re.S).group(1)
        self.Yaxis = re.search(r'Y.axis:\s+(.*?)\n', infile, re.S).group(1)
        self.wdata = int(re.search(r'wdata:\s+(.*?)\n', infile, re.S).group(1))
        self.ldata = int(re.search(r'ldata:\s+(.*?)\n', infile, re.S).group(1))

        # Get field and field units.
        tmp = re.search(r'data\.\.\..*?\n(.*?)\n(.*?)\n', infile, re.S)
        self.fields = tmp.group(1)
        self.field_units = tmp.group(2)

        self.fields = self.fields.split()[1:]
        self.field_units = self.field_units.split()[1:]

        self.xlabel = self.Xaxis + ' [' + self.field_units[0] + ']'
        self.ylabel = self.Yaxis + ' [' + self.field_units[2] + ']'

        # Get data.
        tmp = re.sub(r'#.*?\n', '', infile)
        tmp = re.sub(r'\/\/\n', '', tmp, re.S)
        tmp = tmp.split()
        self.X  = np.array([float(x) for x in tmp[0::4]])
        self.dX = np.array([float(x) for x in tmp[1::4]])
        self.Y  = np.array([float(x) for x in tmp[2::4]])
        self.dY = np.array([float(x) for x in tmp[3::4]])

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

def plot_fn(
    x_eval = None,
    y_eval = None,
    x_exp = None,
    y_exp = None,
    dx_exp = None,
    dy_exp = None,
    outfilename = 'tmp.png',
    xmin = None,
    xmax = None,
    ymin = None,
    ymax = None,
    logscale_x = True,
    logscale_y = True,
    xlabel = '',
    ylabel = ''):

    import matplotlib.pyplot as plt
    fig, (ax0) = plt.subplots(ncols=1)

    if(x_exp != None):
        ax0.errorbar(x_exp, y_exp, xerr=dx_exp, yerr=dy_exp,
            lw=0.5, ecolor='#aaaaaa', fmt='none', marker='',
            capsize=2)

    if(x_eval != None):
        ax0.plot(x_eval, y_eval)

    if(logscale_x):
        ax0.set_xscale('log')

    if(logscale_y):
        ax0.set_yscale('log')

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
    This script is used to read and plot evaluated and experimental data files
    from the EXFOR website.  The files that it reads are from the "See: plotted
    data" link on the upper-right hand part of the EXFOR plot webpage.
    """)

    epilog = textwrap.dedent(
    """
    Typical command line calls might look like:

    > python """ + os.path.basename(__file__) + """ --eval X4sShowData_eval.txt --exp X4sShowData_exp.txt
    """ + u"\u2063")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description, epilog=epilog)

    # Add required, positional, arguments.
    parser.add_argument(
        '--eval', type = str,
        help = 'evaluated cross section file name (e.g., X4sShowData_eval.txt)')
    parser.add_argument(
        '--exp', type = str,
        help = 'experimental data file name (e.g., X4sShowData_exp.txt)')

    parser.add_argument(
        '--prefix', type = str,
        help = 'prefix for naming plots (e.g., zvd_ti46_IRDFF_103)')

    parser.add_argument(
        '--xmin', type = float, default = None,
        help = 'minimum domain value (default: min in all files)')
    parser.add_argument(
        '--xmax', type = float, default = None,
        help = 'maximum domain value (default: max in all files)')
    parser.add_argument(
        '--ymin', type = float, default = None,
        help = 'minimum range value (default: min in all files)')
    parser.add_argument(
        '--ymax', type = float, default = None,
        help = 'maximum range value (default: max in all files)')

    args = parser.parse_args()

    ############################################################
    # Program execution.
    ############################################################

    if(args.eval != None):
        assert(os.path.isfile(args.eval)), 'Input file (' + args.eval + ') not found.'
        data_eval = xs_data_eval(args.eval)

    if(args.exp != None):
        assert(os.path.isfile(args.exp)), 'Input file (' + args.exp + ') not found.'
        data_exp = xs_data_exp(args.exp)

    if(args.eval != None):
        plot_fn(
            x_eval = data_eval.X,
            y_eval = data_eval.Y,
            xlabel = data_eval.xlabel,
            ylabel = data_eval.ylabel,
            xmin = args.xmin,
            xmax = args.xmax,
            ymin = args.ymin,
            ymax = args.ymax,
            logscale_y = False,
            outfilename = args.prefix + '_eval.png'
            )

    if(args.exp != None):
        plot_fn(
            x_exp = data_exp.X,
            y_exp = data_exp.Y,
            dx_exp = data_exp.dX,
            dy_exp = data_exp.dY,
            xmin = args.xmin,
            xmax = args.xmax,
            ymin = args.ymin,
            ymax = args.ymax,
            logscale_y = False,
            xlabel = data_exp.xlabel,
            ylabel = data_exp.ylabel,
            outfilename = args.prefix + '_exp.png'
            )

    if(args.eval != None and args.exp != None):
        plot_fn(
            x_eval = data_eval.X,
            y_eval = data_eval.Y,
            x_exp = data_exp.X,
            y_exp = data_exp.Y,
            dx_exp = data_exp.dX,
            dy_exp = data_exp.dY,
            xlabel = data_eval.xlabel,
            ylabel = data_eval.ylabel,
            xmin = args.xmin,
            xmax = args.xmax,
            ymin = args.ymin,
            ymax = args.ymax,
            logscale_y = False,
            outfilename = args.prefix + '_eval_exp.png'
            )

