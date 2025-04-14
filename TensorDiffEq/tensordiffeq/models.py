import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *
from .fit import *
from tqdm.auto import tqdm, trange
from .output import print_screen


class CollocationSolverND:
    def __init__(self, assimilate=False, verbose=True):
        self.assimilate = assimilate
        self.verbose = verbose

    def compile(self, layer_sizes, f_model, domain, bcs, isAdaptive=False, loss_fn="MSE", optimizer="adam",
                learning_rate=0.005, dict_adaptive=None, init_weights=None, g=None, dist=False):
        """
        Args:
            layer_sizes: A list of layer sizes, can be overwritten via resetting u_model to a keras model
            f_model: PDE definition
            domain: a Domain object containing the information on the domain of the system
            bcs: a list of ICs/BCs for the problem
            isAdaptive: Boolean value determining whether to implement self-adaptive solving
            dict_adaptive: a dictionary with boollean indicating adaptive loss for every loss function
            init_weights: a dictionary with keys "residual" and "BCs". Values must be a tuple with dimension
                          equal to the number of  residuals and boundares conditions, respectively
            g: a function in terms of `lambda` for self-adapting solving. Defaults to lambda^2
            dist: A boolean value determining whether the solving will be distributed across multiple GPUs

        Returns:
            None
        """
        # CHANGED: Set loss function based on loss_type parameter
        if loss_fn == "MSE":
            self.loss_fn = MSE
        elif loss_fn == "MAE":
            self.loss_fn = MAE
        elif loss_fn == "huber":
            self.loss_fn = huber_loss
        elif loss_fn == "sse":
            self.loss_fn = sse_loss
        elif loss_fn == "mean_power":
            self.loss_fn = mean_power_error
        elif loss_fn == "exponential":
            self.loss_fn = exponential_loss
        elif loss_fn == "log_cosh":
            self.loss_fn = log_cosh_loss
        elif loss_fn == "log_mse":
            self.loss_fn = log_mse_loss
        elif loss_fn == "log_huber":
            self.loss_fn = log_huber_loss
        else:
            raise ValueError("Loss function type not recognized.")

        # CHANGED: Set optimizer based on optimizer parameter
        if optimizer.lower() == "adam":
            self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99)
            self.tf_optimizer_weights = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99)
        elif optimizer.lower() == "sgd":
            self.tf_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            self.tf_optimizer_weights = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == "adagrad":
            self.tf_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
            self.tf_optimizer_weights = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer.lower() == "rmsprop":
            self.tf_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            self.tf_optimizer_weights = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == "nadam":
            self.tf_optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            self.tf_optimizer_weights = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        else:
            raise ValueError("Optimizer type not recognized.")

        # self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=.99)
        # self.tf_optimizer_weights = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=.99)
        self.layer_sizes = layer_sizes
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.bcs = bcs
        self.f_model = get_tf_model(f_model)
        self.g = g
        self.domain = domain
        self.dist = dist
        if np.isnan(self.domain.X_f).any():
            raise ValueError("Collocation points contain NaN values!")
        self.X_f_dims = tf.shape(self.domain.X_f)
        self.X_f_len = tf.slice(self.X_f_dims, [0], [1]).numpy()
        # must explicitly cast data into tf.float32 for stability
        self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        self.u_model = neural_net(self.layer_sizes)
        self.batch = None
        self.batch_indx_map = None
        self.lambdas = self.dict_adaptive = self.lambdas_map = None
        self.isAdaptive = isAdaptive

        if self.isAdaptive:
            self.dict_adaptive = dict_adaptive
            self.lambdas, self.lambdas_map = initialize_weights_loss(init_weights)

            if dict_adaptive is None and init_weights is None:
                raise Exception("Adaptive weights selected but no inputs were specified!")
        if (
                self.isAdaptive is False
                and self.dict_adaptive is not None
                and self.lambdas is not None
            ):
            raise Exception(
                "Adaptive weights are turned off but weight vectors were provided. Set the weight vectors to "
                "\"none\" to continue")

    def compile_data(self, x, t, y):
        if not self.assimilate:
            raise Exception(
                "Assimilate needs to be set to 'true' for data assimilation. Re-initialize CollocationSolver1D with "
                "assimilate=True.")
        self.data_x = x
        self.data_t = t
        self.data_s = y

    def update_loss(self):
        loss_bcs = 0.0

        #####################################
        # BOUNDARIES and INIT conditions
        #####################################
        if self.isAdaptive:
            if len(self.lambdas_map['bcs']) > 0:
                idx_lambda_bcs = self.lambdas_map['bcs'][0]

        for counter_bc, bc in enumerate(self.bcs):
            loss_bc = 0.0
            if self.isAdaptive:
                isBC_adaptive = self.dict_adaptive["BCs"][counter_bc]
            else:
                isBC_adaptive = False

            if bc.isPeriodic:
                if isBC_adaptive:
                    raise Exception('TensorDiffEq is currently not accepting Adaptive Periodic Boundaries Conditions')
                else:
                    for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                upper = bc.u_x_model(self.u_model, bc.upper[i])[j][k]
                                lower = bc.u_x_model(self.u_model, bc.lower[i])[j][k]
                                msq = self.loss_fn(upper, lower)  # Use configurable loss function
                                loss_bc = tf.math.add(loss_bc, msq)
            elif bc.isInit:
                if isBC_adaptive:
                    loss_bc = self.loss_fn(self.u_model(bc.input), bc.val, self.lambdas[idx_lambda_bcs])
                    idx_lambda_bcs += 1
                else:
                    loss_bc = self.loss_fn(self.u_model(bc.input), bc.val)
            elif bc.isNeumann:
                if isBC_adaptive:
                    raise Exception('TensorDiffEq is currently not accepting Adaptive Neumann Boundaries Conditions')
                else:
                    for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                target = tf.cast(bc.u_x_model(self.u_model, bc.input[i])[j][k], dtype=tf.float32)
                                msq = self.loss_fn(bc.val, target)
                                loss_bc = tf.math.add(loss_bc, msq)
            elif bc.isDirichlect:
                if isBC_adaptive:
                    loss_bc = self.loss_fn(self.u_model(bc.input), bc.val, self.lambdas[idx_lambda_bcs])
                    idx_lambda_bcs += 1
                else:
                    loss_bc = self.loss_fn(self.u_model(bc.input), bc.val)
            else:
                raise Exception('Boundary condition type is not acceptable')
            loss_bcs = tf.add(loss_bcs, loss_bc)

        #####################################
        # Residual Equations
        #####################################
        if self.n_batches > 1:
            X_batch = []
            for x_in in self.X_f_in:
                indx_on_batch = self.batch_indx_map[self.batch * self.batch_sz:(self.batch + 1) * self.batch_sz]
                X_batch.append(tf.gather(x_in, indx_on_batch))
            f_u_preds = self.f_model(self.u_model, *X_batch)
        else:
            f_u_preds = self.f_model(self.u_model, *self.X_f_in)

        if not isinstance(f_u_preds, tuple):
            f_u_preds = (f_u_preds,)

        loss_res = 0.0
        for counter_res, f_u_pred in enumerate(f_u_preds):
            if self.isAdaptive:
                isRes_adaptive = self.dict_adaptive["residual"][counter_res]
                if isRes_adaptive:
                    idx_lambda_res = self.lambdas_map['residual'][0]
                    lambdas2loss = self.lambdas[idx_lambda_res]
                    if self.n_batches > 1:
                        lambdas2loss = tf.gather(lambdas2loss, indx_on_batch)
                    if self.g is not None:
                        loss_r = self.loss_fn(f_u_pred, constant(0.0), self.g(lambdas2loss))
                    else:
                        loss_r = self.loss_fn(f_u_pred, constant(0.0), lambdas2loss)
                    idx_lambda_res += 1
                else:
                    loss_r = self.loss_fn(f_u_pred, constant(0.0))
            else:
                loss_r = self.loss_fn(f_u_pred, constant(0.0))
            loss_res = tf.math.add(loss_r, loss_res)

        # After computing loss for BCs or residuals
        tf.debugging.check_numerics(loss_bc, message="Boundary condition loss contains NaN!")
        tf.debugging.check_numerics(loss_r, message="Residual loss contains NaN!")
        loss_total = tf.math.add(loss_res, loss_bcs)
        return loss_total

    # @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.update_loss()
            grads = tape.gradient(loss_value, self.variables)

        # Check for NaNs in the gradients before clipping
        for g in grads:
            if g is not None:
                tf.debugging.check_numerics(g, message="Gradient contains NaN!")
        
        # Apply gradient clipping to prevent NaNs
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else None for g in grads]
        
        return loss_value, grads

    def fit(self, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):

        # Can adjust batch size for collocation points, here we set it to N_f
        N_f = self.X_f_len[0]
        self.batch_sz = batch_sz if batch_sz is not None else N_f
        self.n_batches = N_f // self.batch_sz

        if self.isAdaptive and self.dist:
            raise Exception("Currently we dont support distributed training for adaptive PINNs")

        if self.n_batches > 1 and self.dist:
            raise Exception("Currently we dont support distributed minibatching training")

        if self.dist:
            BUFFER_SIZE = len(self.X_f_in[0])
            EPOCHS = tf_iter
            # devices = ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3'],
            try:
                self.strategy = tf.distribute.MirroredStrategy()
            except:
                print(
                    "Looks like we cant find any GPUs available, or your GPUs arent responding to Tensorflow's API. If "
                    "you're receiving this in error, check that your CUDA, "
                    "CUDNN, and other GPU dependencies are installed correctly with correct versioning based on your "
                    "version of Tensorflow")

            print("Number of GPU devices: {}".format(self.strategy.num_replicas_in_sync))

            BATCH_SIZE_PER_REPLICA = self.batch_sz
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync

            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                self.X_f_in).batch(GLOBAL_BATCH_SIZE)

            # self.train_dataset = self.train_dataset.with_options(options)

            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

            start_time = time.time()

            with self.strategy.scope():
                self.u_model = neural_net(self.layer_sizes)
                self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
                self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
                # self.dist_col_weights = tf.Variable(tf.zeros(batch_sz), validate_shape=True)

                if self.isAdaptive:
                    # self.col_weights = tf.Variable(tf.random.uniform([self.batch_sz, 1]))
                    self.u_weights = tf.Variable(self.u_weights)

            fit_dist(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)

        else:
            fit(self, tf_iter=tf_iter, newton_iter=newton_iter, newton_eager=newton_eager)

    # L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(self):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value = self.update_loss()
            grad = tape.gradient(loss_value, self.u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def predict(self, X_star):
        # predict using concatenated data
        u_star = self.u_model(X_star)
        # split data into tuples for ND support
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        # X_star = np.asarray(tmp)
        # X_star = tuple(X_star)
        X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        f_u_star = self.f_model(self.u_model, *X_star)
        return u_star.numpy(), f_u_star.numpy()

    def save(self, path):
        self.u_model.save(path)

    def load_model(self, path, compile_model=False):
        self.u_model = tf.keras.models.load_model(path, compile=compile_model)