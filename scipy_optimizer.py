import numpy as np
from scipy.optimize import minimize
from tensorflow.python import keras
from tensorflow.keras import backend as K

from tqdm import trange, tqdm_notebook

class GradientObserver(keras.optimizers.Optimizer):
    def __init__(self):
        self._vars = []

    def get_updates(self, loss, params):
        self.updates = []
        grads = self.get_gradients(loss, params)
        for param, grad in zip(params, grads):
            shape = K.int_shape(param)
            var = K.zeros(shape)
            self._vars.append(var)
            self.updates.append(K.update_add(var, grad))
        return self.updates

    def get_gradient_values(self):
        values = []
        for v in self._vars:
            values.append(K.eval(v))
        return values

    def clear(self):
        for var in self._vars:
            K.set_value(var, np.zeros(var.shape))


class GeneratorWrapper(keras.utils.Sequence):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
    def __len__(self):
        return 1
    def __getitem__(self, index):
        assert index == 0
        return self._inputs, self._outputs

class ScipyOptimizer(object):
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

        for i in iterator:
            callbacks.on_batch_begin(i)
            inputs, outputs = generator[i]
            batch_cost = self._model.train_on_batch(inputs, outputs)
            if state['verbose']:
                iterator.set_postfix(cost=batch_cost)
            callbacks.on_batch_end(i)
            cost_sum += batch_cost

        generator.on_epoch_end()

        cost = cost_sum
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
        callback_list = keras.callbacks.CallbackList(callbacks)
        state = {
            'epoch': 0,
            'verbose': verbose,
            'callbacks': callback_list,
            'in_epoch': False
        }
        min_options = {
            'maxiter': epochs,
        }

        def on_iteration_end(xk):
            cb = state['callbacks']
            cb.on_epoch_end(state['epoch'])
            state['epoch'] += 1
            state['in_epoch'] = False

        result = minimize(
            self._fun_generator, x0, method=method, jac=True, options=min_options,
            callback=on_iteration_end, args=(generator, state))
        self._update_weights(result['x'])
        return result
