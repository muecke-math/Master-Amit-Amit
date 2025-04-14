import math
import os
import random
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import *
import tensorflow as tf  # required for gradients and random ops

# Set seeds for reproducibility
seed_value = 60
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------
# Domain & Collocation Setup
# ---------------------
def build_domain():
    # Define the domain for x and t (time)
    Domain = DomainND(var=['x', 't'])
    Domain.add('x', [-1.0, 1.0], fidel=1001)
    Domain.add('t', [0.0, 1.0], fidel=1001)
    return Domain

def create_collocation_points(Domain, N_f=100000):
    Domain.generate_collocation_points(N_f)
    return Domain

# ---------------------
# PDE and Boundary Definitions
# ---------------------

# Initial Condition Function
def func_ic(x):
    """
    Defines the initial condition for the problem.
    Given x, returns the function value u(x,0).

    Args:
        x: numpy array of x coordinates.

    Returns:
        u0: numpy array of function values for the initial condition.
    """
    return np.sin(np.pi * x) * np.sin(4 * np.pi * 0)  # u(x,0)

def f_model(u_model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        u = u_model(tf.concat([x, t], 1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)
    u_tt = tape.gradient(u_t, t)
    del tape
    a1 = 1.0
    a2 = 4.0
    ksq = 1.0
    forcing = - (a1 * math.pi)**2 * tf.sin(a1 * math.pi * x) * tf.sin(a2 * math.pi * t) - \
              (a2 * math.pi)**2 * tf.sin(a1 * math.pi * x) * tf.sin(a2 * math.pi * t) + \
              ksq * tf.sin(a1 * math.pi * x) * tf.sin(a2 * math.pi * t)
    f_u = u_xx + u_tt + ksq*u - forcing
    return f_u

# Boundary Condition Function
def build_BCs(domain):
    """
    Constructs and returns a list of boundary and initial condition objects
    for use with CollocationSolverND.compile. It returns:
      1. An initial condition (IC) enforcing u(x,0)=func_ic(x) at time t=0.
      2. A Dirichlet BC on the lower boundary of t (i.e. t = tlower).
      3. A Dirichlet BC on the upper boundary of t (i.e. t = tupper).
      4. A Dirichlet BC on the left boundary of x (i.e. x = xlower).
      5. A Dirichlet BC on the right boundary of x (i.e. x = xupper).

    This function assumes that when adding a variable to DomainND via
    DomainND.add(token, vals, fidel), the dictionary contains keys:
      - token + "linspace" (e.g. "xlinspace")
      - token + "lower" (e.g. "xlower")
      - token + "upper" (e.g. "xupper")
    
    Also, it uses the IC and dirichletBC classes defined in boundaries.py.
    """
    N_b = 100   # Number of points for each spatial boundary
    N_ic = 200  # Number of points for the initial condition

    # Set the time variable if not already defined; assume t is time.
    if domain.time_var is None:
        domain.time_var = "t"

    # Create the initial condition object.
    # The IC class will enforce u(x,0)=func_ic(x) at time t=0.
    ic_bc = IC(domain, [func_ic], [("x",)], n_values=N_ic)

    # Create spatial Dirichlet BC objects using the dirichletBC class.
    # For the t variable (vertical boundaries), "lower" means the lower time boundary and "upper" the upper time boundary.
    bc_t_lower = dirichletBC(domain, tf.zeros((N_b, 1), dtype=tf.float32), var="t", target="lower")
    bc_t_upper = dirichletBC(domain, tf.zeros((N_b, 1), dtype=tf.float32), var="t", target="upper")
    # For the x variable (horizontal boundaries), "lower" corresponds to left and "upper" to right.
    bc_x_lower = dirichletBC(domain, tf.zeros((N_b, 1), dtype=tf.float32), var="x", target="lower")
    bc_x_upper = dirichletBC(domain, tf.zeros((N_b, 1), dtype=tf.float32), var="x", target="upper")

    # Subsample the generated BC inputs so that their shapes match the provided targets.
    def subsample_bc(bc, n_points):
        idx = np.random.choice(bc.input.shape[0], n_points, replace=False)
        bc.input = bc.input[idx]
        bc.val = tf.zeros((n_points, 1), dtype=tf.float32)

    subsample_bc(bc_t_lower, N_b)
    subsample_bc(bc_t_upper, N_b)
    subsample_bc(bc_x_lower, N_b)
    subsample_bc(bc_x_upper, N_b)

    return [ic_bc, bc_t_lower, bc_t_upper, bc_x_lower, bc_x_upper]

# ---------------------
# Model Construction and Compilation
# ---------------------
def build_model(Domain, BCs, N_f):
    # Define the neural network architecture (layer sizes)
    layer_sizes = [2, 50, 50, 50, 50, 1]

    # N_ic = 200  # as used in build_BCs for IC
    # N_b = 100   # as used in build_BCs for each spatial BC
    # dict_adaptive = {"residual": [True],
    #                  "BCs": [True, True, True, True, True]}  # adapt all five loss components
    # init_weights = {"residual": [tf.random.uniform([N_f, 1])],
    #                 "BCs": [100 * tf.random.uniform([N_ic, 1]),
    #                         tf.random.uniform([N_b, 1]),
    #                         tf.random.uniform([N_b, 1]),
    #                         tf.random.uniform([N_b, 1]),
    #                         tf.random.uniform([N_b, 1])]}
    
    model = CollocationSolverND()
    # model.compile(layer_sizes=layer_sizes, f_model=f_model, domain=Domain, bcs=BCs, isAdaptive=True,           # enable adaptive weights
    #               dict_adaptive=dict_adaptive, init_weights=init_weights, loss_fn="MSE", 
    #               optimizer="adam", learning_rate=0.001)
    # mean_power, log_cosh, exponential
    model.compile(layer_sizes=layer_sizes, f_model=f_model, domain=Domain, bcs=BCs,
                  isAdaptive=False, loss_fn="MSE", optimizer="adam", learning_rate=0.005)
    return model

# ---------------------
# Training Function
# ---------------------
def train_model(model):
    model.fit(tf_iter=100, newton_iter=100)
    return model

# ---------------------
# Save and Load Model Functions
# ---------------------
def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "trained_model.keras")
    model.u_model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model, save_dir):
    model_path = os.path.join(save_dir, "trained_model.keras")
    if os.path.exists(model_path):
        model.u_model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"No trained model found at {model_path}. Train the model first.")

# ---------------------
# Plotting Function (Used for both training and testing cases)
# ---------------------
def plot_results(model, Domain, current_directory, Exact_u, baseline_dir, mode='train'):
    x = Domain.domaindict[0]['xlinspace']
    t = Domain.domaindict[1]["tlinspace"]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    # Forward pass for plotting data
    u_pred, f_u_pred = model.predict(X_star)

    # Compute and print the L2 error using high-fidelity data (Exact_u is u_star)
    error_u = tdq.helpers.find_L2_error(u_pred.reshape(1001, 1001), Exact_u)
    if mode == 'train':
        print('Train error u: %e' % (error_u))
    else:
        print('Test error u: %e' % (error_u))
    
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])
    
    plot_folder = os.path.join(current_directory, 'Helmholtz', baseline_dir, mode)
    os.makedirs(plot_folder, exist_ok=True)
    
    tdq.plotting.plot_solution_domain1D(model, [x, t], ub, lb, Exact_u)
    plt.savefig(os.path.join(plot_folder, 'Helmholtz_baseline.png'))
    plt.close()
    
    tdq.plotting.plot_residuals_domain(model, [x, t], f_u_pred, lb, ub)
    plt.savefig(os.path.join(plot_folder, 'Helmholtz_residual.png'))
    plt.close()
    
    tdq.plotting.plot_absolute_error(model, [x, t], Exact_u)
    plt.savefig(os.path.join(plot_folder, 'Helmholtz_absolute_error.png'))
    plt.close()
    
    if model.lambdas is not None:
        tdq.plotting.plot_adaptive_weights(model, scale=1)
        plt.savefig(os.path.join(plot_folder, 'Helmholtz_adaptive_weights.png'))
        plt.close()
    else:
        print("Adaptive weights not used; skipping adaptive weights plot.")

# ---------------------
# High-Fidelity Data Loader
# ---------------------
def load_high_fidelity_data(mode='train'):
    current_directory = os.getcwd()
    # Define the training grid
    x_train = np.linspace(-1.0, 1.0, 1001)
    t_train = np.linspace(0.0, 1.0, 1001)
    xv, tv = np.meshgrid(x_train, t_train)
    Exact_u = np.sin(math.pi*xv)*np.sin(4*math.pi*tv)

    if mode == 'test':
        x_train = np.linspace(-1.0, 1.0, 1001)
        t_train = np.linspace(0.0, 1.0, 1001)
        x_test = np.linspace(-0.95, 0.95, 1001)
        t_test = np.linspace(0.005, 0.995, 1001)
        
        from scipy.interpolate import RectBivariateSpline
        spline = RectBivariateSpline(x_train, t_train, Exact_u)
        Exact_u_test = spline(x_test, t_test)
        noise = np.random.normal(0, 0.01, Exact_u_test.shape)
        Exact_u_test += noise
        return Exact_u_test, current_directory

    return Exact_u, current_directory

# ---------------------
# Main Function with Modes for Training and Testing
# ---------------------
def run_train_and_test():
    # Setup common Domain, BCs, and model
    Domain = build_domain()
    N_f = 100000
    Domain = create_collocation_points(Domain, N_f)
    BCs = build_BCs(Domain)
    model = build_model(Domain, BCs, N_f)
    
    current_directory = os.getcwd()
    baseline_path = os.path.join(current_directory, 'Helmholtz', 'Test')
    os.makedirs(baseline_path, exist_ok=True)
    
    # Training Phase
    print("Running training phase...")
    model = train_model(model)
    Exact_u, current_directory = load_high_fidelity_data(mode='train')
    save_dir = os.path.join(baseline_path, 'model')
    save_model(model, save_dir)
    plot_results(model, Domain, current_directory, Exact_u, baseline_path, mode='train')
    
    # Testing Phase
    print("Running testing phase...")
    model = build_model(Domain, BCs, N_f)
    load_model(model, save_dir)
    Exact_u_test, current_directory_test = load_high_fidelity_data(mode='test')
    plot_results(model, Domain, current_directory_test, Exact_u_test, baseline_path, mode='test')

if __name__ == '__main__':
    run_train_and_test()
