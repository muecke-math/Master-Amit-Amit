
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: mraissi
"""

import numpy as np
import matplotlib as mpl
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np

#mpl.use('pgf')

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": False,  # Change this to False to disable LaTeX rendering
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),
    "pgf.preamble": r"\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}"
}
mpl.rcParams.update(pgf_with_latex)

# I make my own newfig and savefig functions
def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, crop = True):
    if crop == True:
#        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
#        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))

def plot_helmholtz_figure(U_pred, Exact_u, x, y, lb, ub, save_dir):
    """
    Plots the 2D predicted solution U_pred (with colorbar),
    draws vertical lines for slices, and then shows three
    1D slice comparisons between U_pred and Exact_u.

    Parameters
    ----------
    U_pred : 2D numpy array
        PINN-predicted solution on a mesh of shape (len(x), len(y)).
    Exact_u : 2D numpy array
        Exact solution on the same mesh.
    x : 1D numpy array
        x-coordinates for the mesh (length nx).
    y : 1D numpy array
        y-coordinates for the mesh (length ny).
    lb : tuple or list
        Lower bounds of the domain (e.g., [-1.0, -1.0]).
    ub : tuple or list
        Upper bounds of the domain (e.g., [1.0, 1.0]).
    """
    # You can replace this with:
    # fig, ax = newfig(1.3, 1.0)
    # if you want to use your custom "newfig" function from plotting.py.
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Row 0: 2D color map of U_pred
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred,
                  interpolation='nearest',
                  cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # Draw vertical dashed lines at y[250], y[500], y[750]
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(y[250]*np.ones((2,1)), line, 'k--', linewidth=1)
    ax.plot(y[500]*np.ones((2,1)), line, 'k--', linewidth=1)
    ax.plot(y[750]*np.ones((2,1)), line, 'k--', linewidth=1)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('$u(x,y)$', fontsize=10)

    # Row 1: 1D slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    # Slice at index 250
    ax1 = plt.subplot(gs1[0, 0])
    ax1.plot(x, Exact_u[:, 250], 'b-', linewidth=2, label='Exact')
    ax1.plot(x, U_pred[:, 250], 'r--', linewidth=2, label='Prediction')
    ax1.set_xlabel('$y$')
    ax1.set_ylabel('$u(x,y)$')
    ax1.set_title('$y = %.2f$' % (y[250]), fontsize=10)
    ax1.axis('square')
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])

    # Slice at index 500
    ax2 = plt.subplot(gs1[0, 1])
    ax2.plot(x, Exact_u[:, 500], 'b-', linewidth=2, label='Exact')
    ax2.plot(x, U_pred[:, 500], 'r--', linewidth=2, label='Prediction')
    ax2.set_xlabel('$y$')
    ax2.set_ylabel('$u(x,y)$')
    ax2.axis('square')
    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_title('$x = %.2f$' % (y[500]), fontsize=10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
               ncol=5, frameon=False)

    # Slice at index 750
    ax3 = plt.subplot(gs1[0, 2])
    ax3.plot(x, Exact_u[:, 750], 'b-', linewidth=2, label='Exact')
    ax3.plot(x, U_pred[:, 750], 'r--', linewidth=2, label='Prediction')
    ax3.set_xlabel('$y$')
    ax3.set_ylabel('$u(x,y)$')
    ax3.axis('square')
    ax3.set_xlim([-1.1, 1.1])
    ax3.set_ylim([-1.1, 1.1])
    ax3.set_title('$x = %.2f$' % (y[750]), fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()