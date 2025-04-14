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

