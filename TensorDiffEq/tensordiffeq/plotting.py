# Raissi et al plotting scripts - https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py
# All code in this script is credited to Raissi et al


import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf

from scipy.stats import binned_statistic_2d

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


import matplotlib.pyplot as plt

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def plot_solution_domain1D(model, domain, ub, lb, Exact_u=None, u_transpose=False):
    """
    Plot a 1D solution Domain
    Arguments
    ---------
    model : model
        a `model` class which contains the PDE solution
    domain : Domain
        a `Domain` object containing the x,t pairs
    ub: list
        a list of floats containing the upper boundaries of the plot
    lb : list
        a list of floats containing the lower boundaries of the plot
    Exact_u : list
        a list of the exact values of the solution for comparison
    u_transpose : Boolean
        a `bool` describing whether or not to transpose the solution plot of the domain
    Returns
    -------
    None
    """
    X, T = np.meshgrid(domain[0],domain[1])

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    if Exact_u is not None:
        u_star = Exact_u.T.flatten()[:,None]

    u_pred, f_u_pred = model.predict(X_star)
    if u_transpose:
        U_pred = griddata(X_star, u_pred.T.flatten(), (X, T), method='cubic')
    else:
        U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    fig, ax = newfig(1.3, 1.0)

    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    len_ = len(domain[1])//4

    line = np.linspace(domain[0].min(), domain[0].max(), 2)[:,None]
    ax.plot(domain[1][len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][2*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][3*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('u(t,x)', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(domain[0],Exact_u[:,len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title('t = %.2f' % (domain[1][len_]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(domain[0],Exact_u[:,2*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[2*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('t = %.2f' % (domain[1][2*len_]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(domain[0],Exact_u[:,3*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[3*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('t = %.2f' % (domain[1][3*len_]), fontsize = 10)

    plt.show()


def plot_weights(model, scale = 1):
    plt.scatter(model.domain.X_f[:,1], model.domain.X_f[:,0], c = model.lambdas[0].numpy(), s = model.lambdas[0].numpy()/float(scale))
    plt.xlabel(model.domain.domain_ids[1])
    plt.ylabel(model.domain.domain_ids[0])
    plt.show()

# --------------------- NEW PLOTTING FUNCTIONS ---------------------
def plot_residuals_domain(model, domain, f_residual, lb, ub):
    """
    Plot the residual r(x,t) over the spatial-temporal domain.

    Updates:
    ✅ Colormap changed to 'viridis' for better visualization.
    ✅ Ensured proper extent scaling.
    ✅ Added a colorbar for clarity.
    ✅ Labeled axes properly.
    """

    X, T = np.meshgrid(domain[0], domain[1])
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Interpolate the residuals onto the grid
    R = griddata(X_star, f_residual.flatten(), (X, T), method='cubic')

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(R.T, interpolation='nearest', cmap='viridis',
                   extent=[lb[1], ub[1], lb[0], ub[0]],
                   origin='lower', aspect='auto')

    ax.set_xlabel('Time t')
    ax.set_ylabel('Space x')
    ax.set_title('Residual Distribution r(x,t)', fontsize=10)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label='Residual Magnitude')

    # Use tight_layout to prevent cutting off labels
    plt.tight_layout()
    plt.show()

def plot_absolute_error(model, domain, Exact_u):
    """
    Produce a heatmap of the absolute error (|u_pred - u_exact|) 
    with a near-white background for small errors transitioning to blue for larger errors.
    """
    # Unpack domain
    x_vals = domain[0]  # Space (e.g. -1.0 to 1.0)
    t_vals = domain[1]  # Time  (e.g.  0.0 to 1.0)

    # Create a meshgrid for plotting
    X, T = np.meshgrid(x_vals, t_vals)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Predict model output
    u_pred, _ = model.predict(X_star)

    # Reshape predictions onto the grid
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    abs_error = np.abs(U_pred - Exact_u.T)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    # Colormap "YlGnBu" starts near yellowish-white and transitions to blue
    c = ax.imshow(
        abs_error.T, 
        interpolation='nearest',
        cmap='YlGnBu',
        extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()],
        origin='lower',
        aspect='auto'
    )

    ax.set_xlabel('Time t')
    ax.set_ylabel('Space x')
    ax.set_title('u prediction error')

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(c, cax=cax, label='u prediction error')

    # Ensure no labels or colorbar elements are cut off
    plt.tight_layout()
    plt.show()

def plot_adaptive_weights(model, scale=1):
    """
    Plot learned self-adaptive weights across the spatio-temporal domain.

    Updates:
    ✅ Colormap changed to 'viridis' for clearer visualization.
    ✅ Normalized weight values for better color contrast.
    ✅ Higher weight values are emphasized with larger points.
    """

    X_f = model.domain.X_f
    weights = model.lambdas[0].numpy().flatten()

    # Normalize weights for color mapping (0.0 → 1.0)
    w_min, w_max = np.min(weights), np.max(weights)
    norm_weights = (weights - w_min) / (w_max - w_min + 1.0e-12)  # add epsilon to avoid /0

    # Create scatter plot
    plt.figure(figsize=(7, 4))
    sc = plt.scatter(
        X_f[:, 1],  # time on x-axis
        X_f[:, 0],  # space on y-axis
        c=norm_weights,
        s=norm_weights * 30,     # Marker size scales with weight magnitude
        cmap='viridis',
        alpha=0.7,
        edgecolors='none'
    )

    # Labels and title
    plt.xlabel('Time t')
    plt.ylabel('Space x')
    plt.title('Learned Self-Adaptive Weights')

    # Colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Normalized Weight Magnitude')

    plt.tight_layout()
    plt.show()

# def plot_average_adaptive_weights(model):
#     """
#     Plot average learned residue weights over time partitions.

#     Updates:
#     ✅ Matched colors to Fig. 10.
#     ✅ Improved weight scaling to show differences clearly.
#     ✅ X-axis represents training iterations (scaled by 100).
#     ✅ Y-axis shows average weight magnitudes.
#     """

#     # Extract collocation points and final learned weights
#     X_f = model.domain.X_f  # shape (N, 2) -> col 0: x, col 1: t
#     weights = model.lambdas[0].numpy().flatten()  # shape (N,)

#     # Define time partitions
#     t_values = X_f[:, 1]
#     partitions = [0.25, 0.5, 0.75, 1.0]

#     # Prepare storage for partition-wise averages
#     avg_weights = []
#     labels = [
#         r"Residue Weight Average, $t < 0.25$",
#         r"Residue Weight Average, $0.25 \leq t < 0.5$",
#         r"Residue Weight Average, $0.5 \leq t < 0.75$",
#         r"Residue Weight Average, $t \geq 0.75$",
#         "Initial Weight Average"
#     ]
#     colors = ['blue', 'red', 'purple', 'orange', 'green']

#     # Compute average weight in each time partition
#     for i, t_max in enumerate(partitions):
#         if i == 0:
#             mask = (t_values < t_max)
#         else:
#             mask = (t_values >= partitions[i - 1]) & (t_values < t_max)
#         avg_weights.append(np.mean(weights[mask]))

#     # Compute overall (initial) weight average
#     avg_weights.append(np.mean(weights))

#     # Plot the multi-line chart
#     plt.figure(figsize=(7, 5))
#     x_values = np.arange(0, len(avg_weights) * 10, 10)

#     # Each line is just a cumulative sum of a constant value
#     # so it appears as an upward line on the chart
#     for i, avg_weight in enumerate(avg_weights):
#         plt.plot(
#             x_values,
#             np.cumsum(np.ones_like(x_values) * avg_weight),
#             color=colors[i],
#             label=labels[i],
#             linewidth=2
#         )

#     plt.xlabel("Training Iteration x100")
#     plt.ylabel("Average Weight Magnitude")
#     plt.title("Average Learned Residue Weights Over Time Partitions")
#     plt.grid(True)
#     plt.legend(loc="upper left", frameon=True)
#     plt.tight_layout()
#     plt.show()
# --------------------- END NEW FUNCTIONS ---------------------

def plot_glam_values(model, scale = 1):
    plt.scatter(model.t_f, model.x_f, c = model.g(model.col_weights).numpy(), s = model.g(model.col_weights).numpy()/float(scale))
    plt.show()

def plot_residuals(FU_pred, extent):
    fig, ax = plt.subplots()
    ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
                extent=extent,
                origin='lower', aspect='auto')

    #ax.add_collection(ec)
    ax.autoscale_view()
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    cbar = plt.colorbar(ec)
    cbar.set_label('\overline{f}_u prediction')
    plt.show()

def get_griddata(grid, data, dims):
    return griddata(grid, data, dims, method='cubic')
