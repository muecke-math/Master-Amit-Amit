import tensorflow as tf
from tensordiffeq.sampling import LHS
import time as time
import numpy as np


def set_weights(model, w, sizes_w, sizes_b):
    for i, layer in enumerate(model.layers[0:]):
        start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
        end_weights = sum(sizes_w[:i + 1]) + sum(sizes_b[:i])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[i] / sizes_b[i])
        weights = tf.reshape(weights, [w_div, sizes_b[i]])
        biases = w[end_weights:end_weights + sizes_b[i]]
        weights_biases = [weights, biases]
        layer.set_weights(weights_biases)


def get_weights(model):
    w = []
    for layer in model.layers[0:]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)

    w = tf.convert_to_tensor(w)
    return w


def get_sizes(layer_sizes):
    sizes_w = [layer_sizes[i] * layer_sizes[i - 1] for i in range(len(layer_sizes)) if i != 0]
    sizes_b = layer_sizes[1:]
    return sizes_w, sizes_b


def MSE(pred, actual, weights=None):
    if weights is not None:
        return tf.reduce_mean(tf.square(weights * tf.math.subtract(pred, actual)))
    return tf.reduce_mean(tf.square(tf.math.subtract(pred, actual)))

## USELESS LOSSES ##

def MAE(pred, actual, weights=None):
    if weights is not None:
        return tf.reduce_mean(weights * tf.abs(tf.math.subtract(pred, actual)))
    return tf.reduce_mean(tf.abs(tf.math.subtract(pred, actual)))

def huber_loss(pred, actual, delta=1.0, weights=None):
    error = pred - actual
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    loss = tf.where(is_small_error, squared_loss, linear_loss)
    
    if weights is not None:
        return tf.reduce_mean(weights * loss)
    return tf.reduce_mean(loss)

def sse_loss(pred, actual, weights=None):
    """SSE Loss: Sum of Squared Errors"""
    error = pred - actual
    loss = tf.reduce_sum(tf.square(error))
    if weights is not None:
        return tf.reduce_sum(weights * tf.square(error))
    return loss

#####################

# New Loss Functions

def log_mse_loss(pred, actual, weights=None):
    error = pred - actual
    loss = tf.math.log(1 + tf.square(error))
    if weights is not None:
        return tf.reduce_mean(weights * loss)
    return tf.reduce_mean(loss)

def log_huber_loss(pred, actual, delta=1.0, weights=None):
    error = pred - actual
    loss = delta**2 * tf.math.log(1 + (tf.square(error) / delta**2))
    if weights is not None:
        return tf.reduce_mean(weights * loss)
    return tf.reduce_mean(loss)

def log_cosh_loss(pred, actual, weights=None):
    """Log-Cosh Loss: Smoothly penalizes large errors while behaving like MSE for small errors"""
    error = pred - actual
    loss = tf.reduce_mean(tf.math.log(tf.math.cosh(error)))
    if weights is not None:
        return tf.reduce_mean(weights * tf.math.log(tf.math.cosh(error)))
    return loss

###### USELESS ####################

def mean_power_error(pred, actual, p=3, weights=None):
    """Mean Power Error: Penalizes large errors more strongly"""
    error = tf.abs(pred - actual)
    loss = tf.reduce_mean(tf.pow(error, p))
    if weights is not None:
        return tf.reduce_mean(weights * tf.pow(error, p))
    return loss

def exponential_loss(pred, actual, weights=None):
    """Exponential Loss: Exponentially penalizes large errors"""
    error = tf.abs(pred - actual)
    loss = tf.reduce_mean(tf.exp(error) - 1)
    if weights is not None:
        return tf.reduce_mean(weights * (tf.exp(error) - 1))
    return loss
########################################

def g_MSE(pred, actual, g_lam):
    return tf.reduce_mean(g_lam * tf.square(tf.math.subtract(pred, actual)))


def constant(val, dtype=tf.float32):
    return tf.constant(val, dtype=dtype)


def convertTensor(val, dtype=tf.float32):
    return tf.cast(val, dtype=dtype)


def LatinHypercubeSample(N_f, bounds):
    sampling = LHS(xlimits=bounds)
    return sampling(N_f)


def get_tf_model(model):
    return tf.function(model)


def tensor(x, dtype=tf.float32):
    return tf.convert_to_tensor(x, dtype=dtype)


def multimesh(arrs):
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return ans  # returns like np.meshgrid


# if desired, this flattens and hstacks the output dimensions for feeding into a tf/keras type neural network
def flatten_and_stack(mesh):
    dims = np.shape(mesh)
    output = np.zeros((len(mesh), np.prod(dims[1:])))
    for i, arr in enumerate(mesh):
        output[i] = arr.flatten()
    return output.T  # returns in an [nxm] matrix


def initialize_weights_loss(init_weights):
    lambdas = []
    lambdas_map = {}
    counter = 0

    for i, (key, values) in enumerate(init_weights.items()):
        list = []
        for value in values:
            if value is not None:
                lambdas.append(tf.Variable(value, trainable=True, dtype=tf.float32))
                list.append(counter)
                counter += 1
        lambdas_map[key.lower()] = list
    return lambdas, lambdas_map
