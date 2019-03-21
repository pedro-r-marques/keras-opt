# keras-opt
 Keras Scipy optimize interface

Interface between keras optimizers and scipy.optimize. It is used to run full batch optimization rather than mini-batch stochastic gradient descent. It is applicable to factorization of very sparse matrices where stochastic gradient descent is not able to converge.

Example usage:
```
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, use_bias=False, input_dim=4))
model.compile(optimizer=GradientObserver(), loss='mse')

opt = ScipyOptimizer(model)
result, history = opt.fit_generator(generator, epochs=N)

```