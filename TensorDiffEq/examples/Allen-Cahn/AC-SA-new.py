import scipy.io
import math
import os
import random
import tensordiffeq as tdq
# from tensorflow.nn import sigmoid
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
import argparse

# ---------------------
# Domain & Collocation Setup
# ---------------------
def build_domain():
    Domain = DomainND(["x", "t"], time_var='t')
    Domain.add("x", [-1.0, 1.0], 512)
    Domain.add("t", [0.0, 1.0], 201)
    return Domain

def create_collocation_points(Domain, N_f=50000):
    Domain.generate_collocation_points(N_f)
    return Domain

# ---------------------
# PDE and Boundary Definitions
# ---------------------
def func_ic(x):
    return x ** 2 * np.cos(math.pi * x)

def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    return u, u_x

def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_t = tf.gradients(u, t)[0]
    c1 = tdq.utils.constant(0.0001)
    c2 = tdq.utils.constant(5.0)
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u

def build_BCs(Domain):
    init = IC(Domain, [func_ic], var=[['x']])
    x_periodic = periodicBC(Domain, ['x'], [deriv_model])
    return [init, x_periodic]

# ---------------------
# Model Construction and Compilation
# ---------------------
def build_model(Domain, BCs, N_f):
    dict_adaptive = {"residual": [True],
                     "BCs": [True, False]}
    # init_weights = {"residual": [tf.random.uniform([N_f, 1])],
    #                 "BCs": [100 * tf.random.uniform([512, 1]), None]}
    init_weights = {"residual": [tf.random.uniform([N_f, 1], minval=0.01, maxval=0.1)],
                "BCs": [10 * tf.random.uniform([512, 1], minval=0.01, maxval=0.1), None]}
    layer_sizes = [2, 128, 128, 128, 128, 1]
    model = CollocationSolverND()
    model.compile(layer_sizes, f_model, Domain, BCs, isAdaptive=True,
                  dict_adaptive=dict_adaptive, init_weights=init_weights,
                  loss_fn="MSE", optimizer="adam", learning_rate=0.001) # MSE and adam learning_rate0.001(default)
                  # log_cosh, log_mse, log_huber, use learning_rate=1e-3, sgd , adagrad, rmsprop, nadam 
    return model

# ---------------------
# Training Function
# ---------------------
def train_model(model):
    model.fit(tf_iter=10000, newton_iter=10000)
    return model

# ---------------------
# Save and Load Model Functions
# ---------------------
def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "trained_model.keras")  # Add .keras extension

    # Save TensorFlow model inside CollocationSolverND
    model.u_model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model, save_dir):
    model_path = os.path.join(save_dir, "trained_model.keras")  # Use same extension

    if os.path.exists(model_path):
        model.u_model = tf.keras.models.load_model(model_path) #, custom_objects={'sigmoid': sigmoid}) # custom_objects={'sigmoid': sigmoid}
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
    error_u = tdq.helpers.find_L2_error(u_pred.reshape(512, 201), Exact_u)
    if mode == 'train':
        print('Train error u: %e' % (error_u))
    else:
        print('Test error u: %e' % (error_u))
    
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])
    
    # Create a subfolder for the mode (train/test) under the specified baseline directory
    plot_folder = os.path.join(current_directory, 'Allen-Cahn', baseline_dir, mode)
    os.makedirs(plot_folder, exist_ok=True)
    
    # 1. Plot the Solution Domain
    tdq.plotting.plot_solution_domain1D(model, [x, t], ub, lb, Exact_u)
    plt.savefig(os.path.join(plot_folder, 'AC_SA_baseline.png'))
    plt.close()
    
    # 2. Plot the Residual r(x,t) across the domain
    tdq.plotting.plot_residuals_domain(model, [x, t], f_u_pred, lb, ub)
    plt.savefig(os.path.join(plot_folder, 'AC_residual.png'))
    plt.close()
    
    # 3. Plot the Absolute Error between prediction and high-fidelity solution
    tdq.plotting.plot_absolute_error(model, [x, t], Exact_u)
    plt.savefig(os.path.join(plot_folder, 'AC_absolute_error.png'))
    plt.close()
    
    # 4. Plot the Learned Self-Adaptive Weights
    tdq.plotting.plot_adaptive_weights(model, scale=1)
    plt.savefig(os.path.join(plot_folder, 'AC_adaptive_weights.png'))
    plt.close()

# ---------------------
# High-Fidelity Data Loader
# ---------------------
def load_high_fidelity_data(mode='train'):
    current_directory = os.getcwd()
    data = scipy.io.loadmat(os.path.join(current_directory, 'Allen-Cahn', 'AC.mat'))
    Exact = data['uu']
    Exact_u = np.real(Exact)

    if mode == 'test':
        # Define the training grid (assumed from AC.mat dimensions)
        x_train = np.linspace(-1.0, 1.0, 512)
        t_train = np.linspace(0.0, 1.0, 201)

        # Define a slightly shifted test grid
        x_test = np.linspace(-0.95, 0.95, 512)
        t_test = np.linspace(0.005, 0.995, 201)
        
        # Interpolate the high-fidelity data from the training grid to the test grid.
        from scipy.interpolate import RectBivariateSpline
        spline = RectBivariateSpline(x_train, t_train, Exact_u)
        Exact_u_test = spline(x_test, t_test)

        # Optionally, add small noise to the test data to further differentiate it
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
    N_f = 50000
    Domain = create_collocation_points(Domain, N_f)
    BCs = build_BCs(Domain)
    model = build_model(Domain, BCs, N_f)
    
    # Create the baseline directory (inside the 'Allen-Cahn' folder) if it doesn't exist
    current_directory = os.getcwd()
    baseline_path = os.path.join(current_directory, 'Allen-Cahn', 'Baseline') # Define the directory here
    os.makedirs(baseline_path, exist_ok=True)
    
    # ---------------------
    # Training Phase
    # ---------------------
    print("Running training phase...")
    train_model(model)
    Exact_u, current_directory = load_high_fidelity_data(mode='train')
    save_dir = os.path.join(baseline_path, 'model')
    save_model(model, save_dir)  # Save trained model
    plot_results(model, Domain, current_directory, Exact_u, baseline_path, mode='train')
    
    # ---------------------
    # Testing Phase (if you have a saved model, load it instead)
    # ---------------------
    print("Running testing phase...")
    model = build_model(Domain, BCs, N_f)  # Rebuild model before loading weights
    load_model(model, save_dir)  # Load trained model
    Exact_u_test, current_directory_test = load_high_fidelity_data(mode='test')
    plot_results(model, Domain, current_directory_test, Exact_u_test, baseline_path, mode='test')

if __name__ == '__main__':
    # You can now specify a different baseline directory if desired:
    run_train_and_test()