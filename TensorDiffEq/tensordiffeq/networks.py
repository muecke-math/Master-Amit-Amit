import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations


# define the baseline FC neural network model
# information about how to define custom neural networks is available
# in the docs - https://docs.tensordiffeq.io/hacks/networks/index.html
def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,  ##" tanh" YY, "sigmoid" YY, "swish" YY, "softplus" YY.
            kernel_initializer="glorot_normal"))
    model.add(layers.Dense(
        layer_sizes[-1], activation=None,
        kernel_initializer="glorot_normal"))
    return model
