import numpy as np
from scipy.optimize import minimize
from tensorflow import keras
from tensorflow.keras import backend as K # pylint: disable=import-error
from tensorflow.python.keras.callbacks import BaseLogger, CallbackList, History # pylint: disable=no-name-in-module

from tqdm import trange, tqdm_notebook

class GradientObserver(keras.optimizers.Optimizer):
    """
    Implements the Keras Optimizer interface in order to accumulate gradients for
    each mini batch. Gradients are then read at the end of the epoch by the ScipyOptimizer. 
    """
    def __init__(self):
        self._vars = []

    def get_updates(self, loss, params):
        """
        Build the graph nodes that accumulate gradients.
        """
        self.updates = []
        grads = self.get_gradients(loss, params)
        for param, grad in zip(params, grads):
            shape = K.int_shape(param)
            var = K.zeros(shape)
            self._vars.append(var)
            self.updates.append(K.update_add(var, grad))
        return self.updates

    def get_gradient_values(self):
        """
        Read gradient values (at epoch end).
        """
        values = []
        for v in self._vars:
            values.append(K.eval(v))
        return values

    def clear(self):
        """
        Clear gradient values (used at epoch start)
        """
        for var in self._vars:
            K.set_value(var, np.zeros(var.shape))


class GeneratorWrapper(keras.utils.Sequence):
    """
    Converts fit() into fit_generator() interface.
    """
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
    def __len__(self):
        return 1
    def __getitem__(self, index):
        assert index == 0
        return self._inputs, self._outputs

class ScipyOptimizer(object):
    """
    Invokes the underlying model in order to obtain the cost and gradients for the function
    being optimized.
    """
    def __init__(self, model):
        self._model = model
        self._layers = [layer for layer in model._layers if layer.weights]
        self._weights_size = 0
        for layer in self._layers:
            for w in layer.weights:
                self._weights_size += w.shape[0] * w.shape[1]

    def _update_weights(self, x):
        x_offset = 0
        for layer in self._layers:
            w_list = []
            for w in layer.weights:
                shape = w.get_shape()
                w_size = shape[0] * shape[1]
                w_list.append(
                    np.array(x[x_offset:x_offset+w_size]).reshape(shape))
                x_offset += w_size
            layer.set_weights(w_list)
        assert x_offset == self._weights_size

    def _collect_weights(self):
        x_values = np.empty(self._weights_size)
        x_offset = 0
        for layer in self._layers:
            vars = layer.get_weights()
            for var in vars:
                w_size = var.size
                x_values[x_offset:x_offset+w_size] = var.reshape(-1)
                x_offset += w_size
        assert x_offset == self._weights_size
        return x_values

    def _fun_generator(self, x, generator, state):
        self._model.optimizer.clear()
        self._update_weights(x)
        callbacks = state['callbacks']

        if not state['in_epoch']:
            callbacks.on_epoch_begin(state['epoch'])
            state['in_epoch'] = True

        cost_sum = 0
        iterator = trange(len(generator)) if state['verbose'] else range(len(generator))

        state['epoch_logs'] = {}
        epoch_logs = state['epoch_logs']

        for batch_index in iterator:
            inputs, outputs = generator[batch_index]
            batch_logs = {'batch': batch_index, 'size': inputs.shape[0]}
            callbacks.on_batch_begin(batch_index, batch_logs)
            outs = self._model.train_on_batch(inputs, outputs)
            if not isinstance(outs, list):
                outs = [outs]
            for lbl, v in zip(self._model.metrics_names, outs):
                    batch_logs[lbl] = v
                    epoch_logs[lbl] = epoch_logs.get(lbl, 0.0) + v
            callbacks.on_batch_end(batch_index, batch_logs)
            batch_cost = batch_logs['loss']
            if state['verbose']:
                iterator.set_postfix(cost=batch_cost)
            cost_sum += batch_cost

        generator.on_epoch_end()

        # average the metrics
        for lbl in self._model.metrics_names:
            epoch_logs[lbl] = epoch_logs.get(lbl) / len(iterator)

        cost = cost_sum / len(iterator)

        gradients = self._model.optimizer.get_gradient_values()
        x_grad = np.empty(x.shape)
        x_offset = 0
        for grad in gradients:
            w_size = grad.size
            x_grad[x_offset:x_offset + w_size] = grad.reshape(-1)
            x_offset += w_size
        assert x_offset == self._weights_size
        self._cost = cost
        self._gradients = x_grad
        return cost, x_grad

    def fit(self, inputs, outputs, **kwargs):
        return self.fit_generator(GeneratorWrapper(inputs, outputs), **kwargs)

    def fit_generator(self, generator, method='cg', epochs=1, callbacks=None, verbose=True):
        x0 = self._collect_weights()
        history = History()
        _callbacks = [BaseLogger(stateful_metrics=self._model.stateful_metric_names)]
        _callbacks += (callbacks or []) + [history]
        callback_list = CallbackList(_callbacks)
        callback_list.set_model(self._model)
        callback_list.set_params({
            'epochs': epochs,
            'verbose': verbose,
            'metrics': ['loss'],
        })
        state = {
            'epoch': 0,
            'verbose': verbose,
            'callbacks': callback_list,
            'in_epoch': False,
            'epoch_logs': {},
        }
        min_options = {
            'maxiter': epochs,
        }

        def on_iteration_end(xk):
            cb = state['callbacks']
            cb.on_epoch_end(state['epoch'], state['epoch_logs'])
            state['epoch'] += 1
            state['in_epoch'] = False
            state['epoch_logs'] = {}

        callback_list.on_train_begin()
        result = minimize(
            self._fun_generator, x0, method=method, jac=True, options=min_options,
            callback=on_iteration_end, args=(generator, state))
        self._update_weights(result['x'])
        callback_list.on_train_end()
        return result, history
