import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import os
import math
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting import newfig, plot_helmholtz_figure
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import layers
from scipy.interpolate import griddata
from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs

# Set current working directory (if needed)
cwd = os.getcwd()

# ---------------- Architecture Setup ---------------- #
# Define the network architecture (input_dim=2, output_dim=1)
layer_sizes = [2, 50, 50, 50, 50, 1]

# Precompute sizes for weight/bias slicing
sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))

def set_weights(model, w, sizes_w, sizes_b):
    """Set model weights from a flattened vector."""
    for i, layer in enumerate(model.layers):
        start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
        end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[i] / sizes_b[i])
        weights = tf.reshape(weights, [w_div, sizes_b[i]])
        biases = w[end_weights:end_weights + sizes_b[i]]
        layer.set_weights([weights, biases])

def get_weights(model):
    """Flatten and return all model weights as a single tensor."""
    w = []
    for layer in model.layers:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)
    return tf.convert_to_tensor(w)

def neural_net(layer_sizes):
    """Build a fully connected neural network using tanh activations."""
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(width, activation=tf.nn.tanh,
                               kernel_initializer="glorot_normal"))
    model.add(layers.Dense(layer_sizes[-1], activation=None,
                           kernel_initializer="glorot_normal"))
    return model

# Instantiate the PINN model.
u_model = neural_net(layer_sizes)

# ---------------- PDE Definition ---------------- #
def f_model(x, y):
    """
    Compute the PDE residual for a Helmholtz-like equation:
      u_xx + u_yy + k^2 u - forcing = 0
    where the forcing is chosen so that the exact solution is:
      u(x,y) = sin(a1*pi*x)*sin(a2*pi*y)
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        u = u_model(tf.concat([x, y], axis=1))
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    del tape
    a1 = 1.0
    a2 = 4.0
    ksq = 1.0
    forcing = - (a1*math.pi)**2 * tf.sin(a1*math.pi*x) * tf.sin(a2*math.pi*y) \
              - (a2*math.pi)**2 * tf.sin(a1*math.pi*x) * tf.sin(a2*math.pi*y) \
              + ksq * tf.sin(a1*math.pi*x) * tf.sin(a2*math.pi*y)
    f_u = u_xx + u_yy + ksq*u - forcing
    return f_u

# ---------------- Loss Function ---------------- #
def loss(x_f, y_f, x_lb, y_lb, x_ub, y_ub, x_rb, y_rb, x_lftb, y_lftb, col_weights):
    # Compute the PDE residual loss at collocation points.
    f_u_pred = f_model(x_f, y_f)
    # Evaluate u on the boundaries (here we enforce zero Dirichlet)
    u_lb_pred = u_model(tf.concat([x_lb, y_lb], 1))
    u_ub_pred = u_model(tf.concat([x_ub, y_ub], 1))
    u_rb_pred = u_model(tf.concat([x_rb, y_rb], 1))
    u_lftb_pred = u_model(tf.concat([x_lftb, y_lftb], 1))
    mse_b_u = (tf.reduce_mean(tf.square(u_lb_pred)) +
               tf.reduce_mean(tf.square(u_ub_pred)) +
               tf.reduce_mean(tf.square(u_rb_pred)) +
               tf.reduce_mean(tf.square(u_lftb_pred)))
    mse_f_u = tf.reduce_mean(tf.square(col_weights * f_u_pred))
    total_loss = mse_b_u + mse_f_u
    return total_loss, mse_b_u, mse_f_u

def get_loss_and_flat_grad(x_f, y_f, x_lb, y_lb, x_ub, y_ub,
                           x_rb, y_rb, x_lftb, y_lftb, col_weights):
    def loss_and_flat_grad(w):
        with tf.GradientTape() as tape:
            set_weights(u_model, w, sizes_w, sizes_b)
            loss_value, _, _ = loss(x_f, y_f, x_lb, y_lb, x_ub, y_ub,
                                    x_rb, y_rb, x_lftb, y_lftb, col_weights)
        grads = tape.gradient(loss_value, u_model.trainable_variables)
        grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0)
        return loss_value, grad_flat
    return loss_and_flat_grad

# ---------------- Training Function ---------------- #
def fit(x_f, y_f, x_lb, y_lb, x_ub, y_ub,
        x_rb, y_rb, x_lftb, y_lftb, col_weights, tf_iter, newton_iter):
    batch_sz = N_f
    n_batches = N_f // batch_sz
    start_time = time.time()
    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99)
    tf_optimizer_coll = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99)
    print("Starting Adam training...")
    for epoch in range(tf_iter):
        for i in range(n_batches):
            x_f_batch = x_f[i*batch_sz:(i+1)*batch_sz]
            y_f_batch = y_f[i*batch_sz:(i+1)*batch_sz]
            with tf.GradientTape(persistent=True) as tape:
                loss_value, mse_b, mse_f = loss(x_f, y_f, x_lb, y_lb, x_ub, y_ub,
                                                  x_rb, y_rb, x_lftb, y_lftb, col_weights)
                grads = tape.gradient(loss_value, u_model.trainable_variables)
                grads_col = tape.gradient(loss_value, col_weights)
            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
            tf_optimizer_coll.apply_gradients(zip([-grads_col], [col_weights]))
            del tape
            elapsed = time.time() - start_time
            print(f"Iteration: {epoch}, Time: {elapsed:.2f}s, mse_b: {mse_b.numpy()}, mse_f: {mse_f.numpy()}, total loss: {loss_value.numpy()}")
            start_time = time.time()
    print("Starting L-BFGS training...")
    loss_and_flat_grad_func = get_loss_and_flat_grad(x_f, y_f, x_lb, y_lb, x_ub, y_ub,
                                                     x_rb, y_rb, x_lftb, y_lftb, col_weights)
    lbfgs(loss_and_flat_grad_func, get_weights(u_model), Struct(), maxIter=newton_iter, learningRate=0.8)

# ---------------- Prediction Function ---------------- #
def predict(X_star):
    X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
    u_star = u_model(X_star)
    f_u_star = f_model(X_star[:, 0:1], X_star[:, 1:2])
    return u_star.numpy(), f_u_star.numpy()

# ---------------- NTK Computation ---------------- #
def compute_ntk(model, X):
    """
    Compute the Neural Tangent Kernel (NTK) matrix for a given set of inputs X.
    For each input, the flattened gradient of the network output with respect to the trainable parameters
    is computed, and the NTK is the Gram matrix of these gradients.
    """
    def compute_grad(x):
        x = tf.expand_dims(x, axis=0)  # add batch dimension
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y = model(x)
        grads = tape.gradient(y, model.trainable_variables)
        grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0)
        return grad_flat
    grads_tensor = tf.vectorized_map(compute_grad, X)  # shape: [N, num_params]
    ntk = tf.matmul(grads_tensor, grads_tensor, transpose_b=True)
    return ntk

# ---------------- Data Generation ---------------- #
# Domain boundaries
lb = np.array([-1.0])
ub = np.array([1.0])
rb = np.array([1.0])
lftb = np.array([-1.0])

# Number of points
N0 = 200           # initial condition points
N_b = 100          # boundary points (per boundary)
N_f = 100000       # collocation (residue) points

# Initialize collocation weights randomly
col_weights = tf.Variable(tf.random.uniform([N_f, 1]))

# Create a fine mesh for the spatial domain (for visualization)
nx, ny = (1001, 1001)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)

# Create a meshgrid and reshape for ground-truth evaluation
xv, yv = np.meshgrid(x, y)
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
Exact_u = np.sin(math.pi * xv) * np.sin(4 * math.pi * yv)

# Sample initial condition points from the domain
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
# For simplicity, assume initial condition u(x,0)=Exact_u at these points
u0 = Exact_u[idx_x, 0:1]

# Sample boundary points
idx_y = np.random.choice(y.shape[0], N_b, replace=False)
yb = y[idx_y, :]

# Generate collocation points using Latin Hypercube Sampling in 2D
X_f = lb + (ub - lb) * lhs(2, N_f)
x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
y_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)

# Construct boundary point arrays (for four boundaries)
X0   = np.concatenate((x0, 0 * x0), axis=1)               # initial condition: (x0, 0)
X_lb = np.concatenate((yb, 0 * yb + lb[0]), axis=1)         # lower boundary: (x, lb)
X_ub = np.concatenate((yb, 0 * yb + ub[0]), axis=1)         # upper boundary: (x, ub)
X_rb = np.concatenate((0 * yb + rb[0], yb), axis=1)         # right boundary: (rb, y)
X_lftb = np.concatenate((0 * yb + lftb[0], yb), axis=1)      # left boundary: (lftb, y)

# Convert boundary arrays to tensors
x_lb = tf.convert_to_tensor(X_lb[:, 0:1], dtype=tf.float32)
y_lb = tf.convert_to_tensor(X_lb[:, 1:2], dtype=tf.float32)
x_ub = tf.convert_to_tensor(X_ub[:, 0:1], dtype=tf.float32)
y_ub = tf.convert_to_tensor(X_ub[:, 1:2], dtype=tf.float32)
x_rb = tf.convert_to_tensor(X_rb[:, 0:1], dtype=tf.float32)
y_rb = tf.convert_to_tensor(X_rb[:, 1:2], dtype=tf.float32)
x_lftb = tf.convert_to_tensor(X_lftb[:, 0:1], dtype=tf.float32)
y_lftb = tf.convert_to_tensor(X_lftb[:, 1:2], dtype=tf.float32)

# ---------------- Training ---------------- #
# Train using 10k iterations of Adam followed by 10k iterations of L-BFGS
fit(x_f, y_f, x_lb, y_lb, x_ub, y_ub,
    x_rb, y_rb, x_lftb, y_lftb, col_weights,
    tf_iter=10000, newton_iter=10000)

# ---------------- Prediction & Evaluation ---------------- #
# Generate a dense grid for visualization
X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
u_star = Exact_u.flatten()[:, None]

# Get model predictions
u_pred, f_u_pred = predict(X_star)
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Relative L2 Error of u: %e' % (error_u))

# Interpolate predictions onto the grid for plotting
U_pred = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
FU_pred = griddata(X_star, f_u_pred.flatten(), (X, Y), method='cubic')

def test_error_u():
    # Define a coarser test grid for faster evaluation
    x_test = np.linspace(-1, 1, 101)
    y_test = np.linspace(-1, 1, 101)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_star_test = np.hstack((X_test.flatten()[:, None], Y_test.flatten()[:, None]))
    Exact_u_test = np.sin(np.pi * X_test) * np.sin(4 * np.pi * Y_test)
    u_star_test = Exact_u_test.flatten()[:, None]
    u_pred_test, _ = predict(X_star_test)
    error_u_test = np.linalg.norm(u_star_test - u_pred_test, 2) / np.linalg.norm(u_star_test, 2)
    print("Test Relative L2 Error: %e" % error_u_test)
    return error_u_test

test_error_u()

# ---------------- NTK Analysis ---------------- #
# For analysis, compute the NTK matrix on a small subset of inputs.
N_ntk = 10
X_ntk = lhs(2, N_ntk)
X_ntk = tf.convert_to_tensor(X_ntk, dtype=tf.float32)
ntk_matrix = compute_ntk(u_model, X_ntk)
print("Computed NTK matrix shape:", ntk_matrix.shape)
print("NTK matrix:\n", ntk_matrix.numpy())

# # ---------------- Plotting Example ---------------- #
# # Plot the predicted solution U_pred
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.axis('off')
# gs0 = gridspec.GridSpec(1, 2)
# gs0.update(top=0.94, bottom=0.33, left=0.15, right=0.85, wspace=0)
# ax0 = plt.subplot(gs0[:, :])
# h = ax0.imshow(U_pred, interpolation='nearest', cmap='YlGnBu',
#                extent=[lb[0], ub[0], lb[0], ub[0]],
#                origin='lower', aspect='auto')
# divider = make_axes_locatable(ax0)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
# ax0.set_xlabel('$x$')
# ax0.set_ylabel('$y$')
# ax0.set_title('$u(x,y)$', fontsize=10)
# plt.show()

######################################################################
############################# Plotting ###############################
######################################################################

fig, ax = newfig(1.3, 1.0)
ax.axis('off')

####### Row 0: h(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

# h = ax.imshow(U_pred, interpolation='nearest', cmap='YlGnBu',
#               extent=[lb[1], ub[1], lb[0], ub[0]],
#               origin='lower', aspect='auto')
h = ax.imshow(U_pred, interpolation='nearest', cmap='YlGnBu',
              extent=[lftb[0], rb[0], lb[0], ub[0]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)


line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(y[250]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(y[500]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(y[750]*np.ones((2,1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
leg = ax.legend(frameon=False, loc = 'best')

ax.set_title('$u(x,y)$', fontsize = 10)

####### Row 1: h(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_u[:,250], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[:,250], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$y$')
ax.set_ylabel('$u(x,y)$')
ax.set_title('$y = %.2f$' % (y[250]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_u[:,500], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[:,500], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$y$')
ax.set_ylabel('$u(x,y)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$x = %.2f$' % (y[500]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_u[:,750], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[:,750], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$y$')
ax.set_ylabel('$u(x,y)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$x = %.2f$' % (y[750]), fontsize = 10)

#plot prediction
fig, ax = plt.subplots()

ec = plt.imshow(U_pred, interpolation='nearest', cmap='rainbow',
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')


ax.autoscale_view()
ax.set_xlabel('$y$')
ax.set_ylabel('$x$')
cbar = plt.colorbar(ec)
cbar.set_label('$u(x,y)$')
plt.title("Predicted $u(x,y)$",fontdict = {'fontsize': 14})
plt.show()


#show f_u_pred, we want this to be ~0 across the whole domain
fig, ax = plt.subplots()

ec = plt.imshow(FU_pred, interpolation='nearest', cmap='rainbow',
            extent=[-1.0, 1, -1.0, 1.0],
            origin='lower', aspect='auto')

#ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
cbar = plt.colorbar(ec)
cbar.set_label('$\overline{f}_u$ prediction')
plt.savefig(os.path.join(cwd, 'Helmholtz/Baseline_NTK_new/U_pred_plot.png'))
# plt.show()

#plot prediction error
fig, ax = plt.subplots()

ec = plt.imshow((U_pred - Exact_u), interpolation='nearest', cmap='YlGnBu',
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')

#ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
cbar = plt.colorbar(ec)
cbar.set_label('$u$ prediction error')
# plt.title("Prediction Error",fontdict = {'fontsize': 14})
plt.savefig(os.path.join(cwd, 'Helmholtz/Baseline_NTK_new/FU_pred_plot.png'))
# plt.show()

# # print collocation point weights
# plt.scatter(x_f, y_f, c = col_weights.numpy(), s = col_weights.numpy()/100)
# plt.savefig(os.path.join(cwd, 'Helmholtz/he_NTK_collocation_weights.png'))
# # plt.show()

############## plotting solution ########################

plot_helmholtz_figure(U_pred=U_pred,
                      Exact_u=Exact_u,
                      x=x,
                      y=y,
                      lb=[-1.0, -1.0],
                      ub=[1.0, 1.0],
                      save_dir=os.path.join(cwd, 'Helmholtz/Baseline_NTK_new/baseline_ntk.png'))
