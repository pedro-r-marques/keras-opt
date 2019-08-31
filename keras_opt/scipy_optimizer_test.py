from __future__ import print_function

import random
import unittest

import numpy as np
import numpy.testing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Concatenate, Embedding, Dense, Dot, Input, Lambda  # pylint: disable=import-error
from tensorflow.keras.models import Sequential, Model  # pylint: disable=import-error
from tensorflow.keras import backend as K  # pylint: disable=import-error

from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from keras_opt.scipy_optimizer import ScipyOptimizer, GradientObserver


class test_generator(keras.utils.Sequence):
    def __init__(self, matrix, batch_size=32):
        self._matrix = matrix
        self._batch_size = batch_size
        self._block_size = (matrix.size - 1) / batch_size + 1

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self._matrix.size == 0:
            return 0
        return int((self._matrix.size - 1) / self._batch_size) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        start = index * self._batch_size
        end = min(start + self._batch_size, self._matrix.size)
        X = np.empty((end - start, 2))
        X[:, 0] = self._matrix.row[start:end]
        X[:, 1] = self._matrix.col[start:end]
        y = self._matrix.data[start:end]
        return X, y


def make_test_matrix(shape, datapoints):
    matrix = dok_matrix(shape)
    for _ in range(datapoints):
        while True:
            row = random.randint(0, shape[0] - 1)
            col = random.randint(0, shape[1] - 1)
            if matrix.get((row, col)):
                continue
            value = 1
            if row >= shape[0] / 2:
                value += 2
            if col >= shape[1] / 2:
                value += 1
            matrix[row, col] = value
            break
    return matrix


def make_embedding_model(shape, embedding_size):
    coordinates = Input(shape=(2,), dtype=tf.int32)
    row_embedding = Embedding(shape[0], embedding_size, input_length=1)
    col_embedding = Embedding(shape[1], embedding_size, input_length=1)
    row = Lambda(lambda x: x[:, 0])(coordinates)
    col = Lambda(lambda x: x[:, 1])(coordinates)
    row_vecs = row_embedding(row)
    col_vecs = col_embedding(col)
    y_r = Dot(1)([row_vecs, col_vecs])
    model = Model(inputs=coordinates, outputs=y_r)
    model.compile(optimizer=GradientObserver(), loss='mean_squared_error')
    return model


class ScipyOptimizerTest(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

    def test_lr(self):
        model = Sequential()
        model.add(Dense(1, use_bias=False, input_dim=4))
        model.compile(optimizer=GradientObserver(), loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return 4*a + 2*b + 3*c + d

        inputs = np.random.rand(10, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        opt = ScipyOptimizer(model)
        result, hist = opt.fit(inputs, outputs, epochs=30, verbose=False)
        self.assertTrue(result['success'])
        self.assertTrue('loss' in hist.history)

        layers = [layer for layer in model._layers if layer.weights]
        w = layers[0].get_weights()[0].reshape(-1)
        w_p = opt._collect_weights()
        numpy.testing.assert_almost_equal(w, w_p)
        numpy.testing.assert_almost_equal(w, [4.0, 2.0, 3.0, 1.0], decimal=4)

    def test_2layer(self):
        model = Sequential()
        model.add(Dense(3, use_bias=False, input_dim=4))
        model.add(Dense(1, use_bias=False))
        model.compile(optimizer=GradientObserver(), loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return a*b + 2*b + 3*c + d

        inputs = np.random.rand(10, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        opt = ScipyOptimizer(model)
        opt.fit(inputs, outputs, epochs=15, verbose=False)

        pred = model.predict(inputs)
        delta = outputs - pred.reshape(-1)
        self.assertEqual(delta.shape, outputs.shape)
        self.assertLess(delta.sum(), 0.01)

    def test_fit_generator(self):
        matrix = make_test_matrix((10, 10), 50)
        generator = test_generator(matrix.tocoo())
        model = make_embedding_model(matrix.shape, 3)
        opt = ScipyOptimizer(model)
        result, _ = opt.fit_generator(generator, epochs=200, verbose=False)
        self.assertLess(result['fun'], 1.0e-3)

    def test_1dim(self):
        def test_fn(x):
            if x > 0.5:
                return 1
            return 0

        def make_model():
            inp = Input(shape=(1,))
            h_layer = Dense(1,
                            kernel_initializer=keras.initializers.RandomUniform(
                                0.0, 1.0),
                            activation='relu')(inp)
            outp = Dense(1, activation='sigmoid')(h_layer)
            return Model(inp, outp)

        model = make_model()
        model.compile(optimizer=GradientObserver(), loss='mse')
        opt = ScipyOptimizer(model)
        X = np.random.rand(100)
        y = np.vectorize(test_fn)(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

        result, _ = opt.fit(X_train, y_train, epochs=20, verbose=False)
        self.assertTrue(result['success'])
        self.assertLessEqual(model.test_on_batch(X_test, y_test), 1.0e-5)

    def test_val_data(self):
        def test_fn(x):
            if x > 0.8:
                return 2
            return 0

        def make_model():
            inp = Input(shape=(1,))
            h_layer = Dense(1,
                            kernel_initializer=keras.initializers.RandomUniform(
                                0.0, 1.0),
                            activation='relu')(inp)
            outp = Dense(1, activation='sigmoid')(h_layer)
            return Model(inp, outp)

        model = make_model()
        model.compile(optimizer=GradientObserver(),
                      loss='mse', metrics=['mae'])
        opt = ScipyOptimizer(model)
        X = np.random.rand(200)
        y = np.vectorize(test_fn)(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

        result, hist = opt.fit(X_train, y_train, epochs=50,
                               validation_data=(X_test, y_test), verbose=False)
        self.assertLessEqual(result['fun'], 0.2)
        print(hist.history.keys())
        self.assertTrue('val_loss' in hist.history)
        self.assertTrue('val_mean_absolute_error' in hist.history or
                        'val_mae' in hist.history)

    def test_mult_inputs(self):
        def test_fn(x, y):
            return 2.0 * x + 4.0 * y + 1.0

        def make_model():
            x = Input(shape=(1, ))
            y = Input(shape=(1, ))
            join = Concatenate()([x, y])
            z = Dense(1)(join)
            return Model([x, y], z)

        model = make_model()
        model.compile(optimizer=GradientObserver(), loss='mse')
        opt = ScipyOptimizer(model)
        X = np.random.rand(10)
        Y = np.random.rand(10)
        Z = np.vectorize(test_fn)(X, Y)

        result, _ = opt.fit([X, Y], Z, epochs=100, verbose=False)
        self.assertTrue(result['success'])

    def test_non_trainable(self):
        """BatchNormalization uses non-trainable weights.
        """
        model = Sequential()
        model.add(Dense(3, use_bias=False, input_dim=4))
        model.add(BatchNormalization())
        model.add(Dense(1, use_bias=False))
        model.compile(optimizer=GradientObserver(), loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return a*b + 2*b + 3*c + d

        inputs = np.random.rand(10, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        opt = ScipyOptimizer(model)
        result, _ = opt.fit(inputs, outputs, epochs=50, verbose=False)
        self.assertLessEqual(result['fun'], 1.0e3)
        self.assertEqual(result['status'], 2, result['message'])


if __name__ == '__main__':
    unittest.main()
