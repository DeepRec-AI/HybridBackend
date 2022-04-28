# Advanced Features

## 1. Configuration

### 1.1 APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.Context
    :members:
    :special-members: __init__
.. autofunction:: hybridbackend.tensorflow.function
.. autofunction:: hybridbackend.tensorflow.scope
```

### 1.2 Example: Get rank and world size

```python
import hybridbackend.tensorflow as hb

print(f'{hb.context.rank}-of-{hb.context.world_size}')
```

### 1.3 Example: Update options globally

```python
hb.context.options.emb_backend = 'MYEMB'
```

or

```bash
HB_EMB_BACKEND=MYEMB python xxx.py
```

### 1.4 Example: Update options inside a function

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.function(emb_device='/cpu:0')
def foo():
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

### 1.5 Example: Update options within a scope

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

with hb.scope(emb_device='/cpu:0'):
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

### 1.6 Example: Use user-defined embedding backend

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

class MyEmbeddingBackend(hb.feature_column.EmbeddingBackend):
  NAME = 'MYEMB'

  def build(self, column, name, shape, **kwargs):
    return mymodule.get_my_emb(name, shape, **kwargs)

  def lookup(self, column, weight, inputs, sharded=False, buffered=False):
    r'''Lookup for embedding vectors.
    '''
    return mymodule.lookup(weight, inputs)

hb.feature_column.EmbeddingBackend.register(MyEmbeddingBackend())
```

```python
from mymodule import MyEmbeddingBackend

hb.context.options.emb_backend = 'PAIEV'
```

## 2. Customization

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.wraps
```

### 2.2 Example: Defines a new optimizer

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

### 2.3 Example: Wraps an existing estimator

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

estimator = hb.wraps(MyEstimator)(model_fn=model_fn)
```

## 3. Nodes with Multiple GPUs

For nodes with more than 1 GPUs, HybridBackend provides a launcher
`hybridbackend.run` which reads environment variable `NVIDIA_VISIBLE_DEVICES` or
`CUDA_VISIBLE_DEVICES` to generate single-GPU workers.

Example:

```bash
# Launch workers for each GPU by reading environment variable
# `NVIDIA_VISIBLE_DEVICES`.
python -m hybridbackend.run python /path/to/main.py
```
