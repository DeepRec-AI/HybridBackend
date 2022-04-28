# Model Definition

## 1. Configuration

### 1.1 APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.Context
    :members:
    :special-members: __init__
.. autofunction:: hybridbackend.tensorflow.function
.. autofunction:: hybridbackend.tensorflow.scope
```

### 1.2 Examples

Case 1: Get rank and world size

```python
print(f'{hb.context.rank}-of-{hb.context.world_size}')
```

Case 2: Configure options globally

```python
hb.context.options.emb_backend = 'MYEMB'
```

or

```bash
HB_EMB_BACKEND=MYEMB python xxx.py
```

Case 3: Configure options inside a function

```python
@hb.function(emb_device='/cpu:0')
def foo():
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

Case 3: Configure options within a scope

```python
with hb.scope(emb_device='/cpu:0'):
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

## 2. Customized Optimizers

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.wraps
```

### 2.2 Examples

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.wraps
class MyOptimizer(tf.train.Optimizer):
  # ...

# ...
def foo():
  loss = tf.losses.get_total_loss()
  opt = MyOptimizer(learning_rate=lr)
  train_op = opt.minimize(loss)
```
