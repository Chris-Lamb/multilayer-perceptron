import numpy as np
from utils import ACTIVATIONS, DERIVATIVES, LOSSES
import sys


class MLP(object):

    def __init__(self, hidden_layer_sizes=(256, 128),
                 activation='relu',
                 batch_size=100,
                 shuffle_batches=True,
                 annealing=True,
                 annealing_coef=10,
                 learning_rate_init=0.001,
                 initial_weights='normal',
                 momentum=True,
                 alpha=0.9,
                 nesterov_momentum=True,
                 mu=0.5,
                 max_iter=100,
                 tol=1.0e-6,
                 verbose=False,
                 early_stopping=True,
                 validation_fraction=0.2,
                 shuffle_validation=True,
                 fit_biases=True):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_layers = len(self.hidden_layer_sizes) + 2
        self.activation = activation
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.annealing = annealing
        self.annealing_coef = annealing_coef
        self.learning_rate_init = learning_rate_init
        self.initial_weights = initial_weights
        self.momentum = momentum
        self.alpha = alpha
        self.nesterov_momentum = nesterov_momentum
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.shuffle_validation = shuffle_validation
        self.fit_biases = fit_biases
        self.weights = None
        self.biases = None
        self.delta_weights = None
        self.delta_biases = None
        self.n_outputs = None
        self.n_iter = 0
        self.loss_curve = None
        self.accuracy_curve = None
        self.val_loss_curve = None
        self.val_accuracy_curve = None

    def predict(self, X):

        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_widths = [X.shape[1]] + hidden_layer_sizes + [self.n_outputs]

        activations = [X]
        for i in range(self.n_layers - 1):
            activations.append(np.empty((X.shape[0], layer_widths[i + 1])))

        self._forward(activations)

        return [np.argmax(y) for y in activations[-1]]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.sum([pred == np.argmax(target) for pred, target
                       in zip(predictions, y)]) / (1.0 * y.shape[0])

    def fit(self, X, y):

        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        n_samples, n_features = X.shape

        self.n_outputs = y.shape[1]
        # Fan in/ Fan out of each layer
        layer_widths = ([n_features] + hidden_layer_sizes + [self.n_outputs])

        self._initialize(y, layer_widths)

        if self.early_stopping:
            X, X_val, y, y_val = self._split_validation(X, y)
            self.val_loss_curve = [self._compute_loss(X_val, y_val)]
            self.val_accuracy_curve = [self.score(X_val, y_val)]
            # Re-evaluate number of samples after split
            n_samples = X.shape[0]

        self.loss_curve.append(self._compute_loss(X, y))
        self.accuracy_curve.append(self.score(X, y))

        batch_size = np.clip(self.batch_size, 1, n_samples)

        for epoch in range(self.max_iter):

            idx = list(range(n_samples))
            if self.shuffle_batches:
                idx = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):

                if self.nesterov_momentum:
                    self._update_nesterov()

                X_batch = X[idx[i:i+batch_size]]
                y_batch = y[idx[i:i+batch_size]]

                n_samples_batch = X_batch.shape[0]

                activations = [X_batch]
                activations.extend(np.empty((n_samples_batch, fan_out))
                                   for fan_out in layer_widths[1:])

                deltas = [np.empty_like(layer) for layer in activations]

                weight_grads = [np.empty_like(weight) for weight in self.weights]
                bias_grads = [np.empty_like(bias) for bias in self.biases]

                self._backprop(X_batch, y_batch, activations, deltas, weight_grads, bias_grads)

                self._update_weights(weight_grads, bias_grads)

            self.n_iter += 1

            self.loss_curve.append(self._compute_loss(X, y))
            self.accuracy_curve.append(self.score(X, y))

            verbose_str = 'Epoch: {}'.format(self.n_iter)
            verbose_str += ' Train Loss: {:.4f}'.format(self.loss_curve[-1])
            verbose_str += ' Train Accuracy: {:.4f}'.format(self.accuracy_curve[-1])

            if self.early_stopping:

                self.val_loss_curve.append(self._compute_loss(X_val, y_val))
                self.val_accuracy_curve.append(self.score(X_val, y_val))

                verbose_str += ' Val Loss: {:.4f}'.format(self.val_loss_curve[-1])
                verbose_str += ' Val Accuracy: {:.4f}'.format(self.val_accuracy_curve[-1])

            if self.verbose:
                sys.stdout.write('\r' + verbose_str)
                sys.stdout.flush()

            if self._stop_conditions_triggered():
                break

    def _initialize(self, y, layer_widths):

        self.n_iter = 0
        self.n_outputs = y.shape[1]
        self.n_layers = len(layer_widths)
        self.loss_curve = []
        self.accuracy_curve = []
        self.weights = []
        self.biases = []

        for i in range(self.n_layers - 1):
            weight_init, bias_init = self._init_weight(layer_widths[i], layer_widths[i + 1])
            self.weights.append(weight_init)
            self.biases.append(bias_init)

        self.delta_weights = [np.zeros(weights.shape, dtype=float) for weights in self.weights]

        self.delta_biases = [np.zeros(bias.shape, dtype=float) for bias in self.biases]

    def _init_weight(self, fan_in, fan_out):

        if self.initial_weights == 'normal':
            mean = 0.0
            variance = 1/np.sqrt(fan_in)

            weight_init = np.random.normal(mean, variance, (fan_in, fan_out))

            if self.fit_biases:
                bias_init = np.random.normal(mean, variance, fan_out)
            else:
                bias_init = np.zeros(fan_out, dtype=float)

        elif self.initial_weights == 'uniform':
            bounds = 0.001

            weight_init = np.random.uniform(-bounds, bounds, (fan_in, fan_out))

            if self.fit_biases:
                bias_init = np.random.uniform(-bounds, bounds, fan_out)
            else:
                bias_init = np.zeros(fan_out, dtype=float)

        else:
            raise ValueError

        return weight_init, bias_init

    def _split_validation(self, X, y):

        n_samples = X.shape[0]

        idx = list(range(n_samples))
        if self.shuffle_validation:
            idx = np.random.permutation(n_samples)

        X_val = X[idx[-int(n_samples*self.validation_fraction):]]
        y_val = y[idx[-int(n_samples*self.validation_fraction):]]

        X = X[idx[:-int(n_samples*self.validation_fraction)]]
        y = y[idx[:-int(n_samples*self.validation_fraction)]]

        return X, X_val, y, y_val

    def _forward(self, activations):

        inputs = [np.empty_like(act) for act in activations]

        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers - 1):
            inputs[i+1] = np.dot(activations[i], self.weights[i])
            inputs[i+1] += self.biases[i]

            if (i + 1) != (self.n_layers - 1):
                activations[i + 1] = hidden_activation(inputs[i + 1])

        output_activation = ACTIVATIONS['softmax']
        activations[-1] = output_activation(inputs[-1])

        return inputs

    def _backprop(self, X, y, activations, deltas, weight_grads, bias_grads):

        n_samples = X.shape[0]
        # Forward propagate
        inputs = self._forward(activations)
        # Backward propagate
        last = self.n_layers - 2
        deltas[last] = activations[-1] - y
        # Compute gradient for the last layer
        self._compute_loss_gradient(last, n_samples, activations, deltas, weight_grads, bias_grads)

        for i in range(self.n_layers - 2, 0, -1):
            deltas[i-1] = np.dot(deltas[i], self.weights[i].T)
            deltas[i-1] *= DERIVATIVES[self.activation](inputs[i], y=activations[i])

            self._compute_loss_gradient(i - 1, n_samples, activations, deltas, weight_grads,
                                        bias_grads)

        return weight_grads, bias_grads

    def _compute_loss_gradient(self, layer, n_samples, activations, deltas, weight_grads,
                               bias_grads):

        weight_grads[layer] = np.dot(activations[layer].T, deltas[layer])
        bias_grads[layer] = np.mean(deltas[layer], axis=0)

        return weight_grads, bias_grads

    def _compute_loss(self, X, y):

        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        n_samples, n_features = X.shape

        self.n_outputs = y.shape[1]
        # Fan in/ Fan out of each layer
        layer_widths = ([n_features] + hidden_layer_sizes + [self.n_outputs])

        activations = [X]
        activations.extend(np.empty((n_samples, fan_out)) for fan_out in layer_widths[1:])
        self._forward(activations)
        loss = LOSSES['cross_entropy'](activations[-1], y)

        return loss

    def _update_nesterov(self):

        self.weights = [weight + self.mu * delta for weight, delta
                        in zip(self.weights, self.delta_weights)]

        if self.fit_biases:

            self.biases = [bias + self.mu * delta for bias, delta
                           in zip(self.biases, self.delta_biases)]

    def _update_weights(self, weight_grads, bias_grads):

        eta = self.learning_rate_init
        if self.annealing:
            eta /= (1 + (self.n_iter / self.annealing_coef))

        coef = 0.0

        if self.momentum:
            coef = self.alpha

        if self.nesterov_momentum:
            coef = self.mu

        self.delta_weights = [coef * weight - eta * grad for weight, grad in
                              zip(self.delta_weights, weight_grads)]

        self.weights = [weight + delta for weight, delta in zip(self.weights, self.delta_weights)]

        if self.fit_biases:

            self.delta_biases = [coef * bias - eta * grad for bias, grad in
                                 zip(self.delta_biases, bias_grads)]

            self.biases = [bias + delta for bias, delta in zip(self.biases, self.delta_biases)]

    def _stop_conditions_triggered(self):

        if self.n_iter <= 5:
            return False

        train_improvement = any([self.loss_curve[-i] < self.loss_curve[-i-1] -
                                 self.tol for i in range(1, 3)])

        if not train_improvement:
            return True

        if self.early_stopping:
            val_improvement = any([self.val_loss_curve[-i] < self.val_loss_curve[-i-1] -
                                   self.tol for i in range(1, 3)])

            if not val_improvement:
                return True

        return False
