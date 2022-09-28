# Distributed Training

## 1. Cluster Configuration

HybridBackend provides `hb.context` for cluster configurtion. For nodes with
more than 1 GPUs, HybridBackend provides a launcher `hybridbackend.run` which
reads environment variable `NVIDIA_VISIBLE_DEVICES` or
`CUDA_VISIBLE_DEVICES` to generate multiple workers on each GPU.

### 1.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.run
.. autofunction:: hybridbackend.tensorflow.context
.. autoclass:: hybridbackend.tensorflow.Context
    :members:
    :special-members: __init__
```

### 1.2 Example: Get rank and world size

```python
import hybridbackend.tensorflow as hb

print(f'{hb.context.rank}-of-{hb.context.world_size}')
```

### 1.3 Example: Update options globally

```python
hb.context.options.grad_nbuckets = 2
```

or

```bash
HB_GRAD_NBUCKETS=2 python xxx.py
```

### 1.4 Example: Launch workers on multiple GPUs

```bash
# Launch workers for each GPU by reading environment variable
# `NVIDIA_VISIBLE_DEVICES`.
python -m hybridbackend.run python /path/to/main.py
```

## 2. Data Parallelism

HybridBackend provides `hb.functions` and `hb.scope` to rewrite variables and
optimizers for supporting data paralleism. Also `hb.wraps` could be used for
user-defined optimizers.

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.function
.. autofunction:: hybridbackend.tensorflow.scope
.. autofunction:: hybridbackend.tensorflow.wraps
```

### 2.2 Example: Training inside a function

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.function(grad_nbuckets=2)
def foo():
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

### 2.3 Example: Training within a scope

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

with hb.scope(grad_nbuckets=2):
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

### 2.4 Example: Defines a new optimizer

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

## 2. Model Parallelism

HybridBackend provides option `sharding` to shard variables and support
model-parallel embedding layers.

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.data.make_one_shot_iterator
.. autofunction:: hybridbackend.tensorflow.data.make_initializable_iterator
.. autofunction:: hybridbackend.tensorflow.train.export
.. autofunction:: hybridbackend.tensorflow.metrics.accuracy
.. autofunction:: hybridbackend.tensorflow.metrics.auc
```

### 2.2 Example: Sharding inside a function

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.function(grad_nbuckets=2)
def foo():
  # ...
  with hb.scope(sharding=True):
    embedding_weights = tf.get_variable(
      'emb_weights', shape=[bucket_size, dim_size])
  embedding = tf.nn.embedding_lookup(embedding_weights, ids)
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

### 2.3 Example: Evaluation

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

def eval_fn():
  # ...
  auc_and_update = hb.metrics.auc(
    labels=eval_labels,
    predictions=eval_logits)
  return {'auc': auc_and_update}

with tf.Graph().as_default():
  with hb.scope():
    batch = hb.data.make_one_shot_iterator(train_ds).get_next()
    # ...
    with hb.scope(sharding=True):
      embedding_weights = tf.get_variable(
        'emb_weights', shape=[bucket_size, dim_size])
    embedding = tf.nn.embedding_lookup(embedding_weights, ids)
    # ...

    with tf.train.MonitoredTrainingSession(
        '',
        hooks=hooks,
        eval_every_n_iter=1000,
        eval_fn=eval_fn) as sess:
      while not sess.should_stop():
        sess.run(train_op)
```

### 2.4 Example: Exporting to SavedModel

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Export trained models.
def _on_export():
  example_spec = {}
  for f in numeric_fields:
    example_spec[f] = tf.io.FixedLenFeature([1], dtype=tf.int64)
  for f in categorical_fields:
    example_spec[f] = tf.io.VarLenFeature(dtype=tf.int64)

  serialized_examples = tf.placeholder(
      dtype=tf.string,
      shape=[None],
      name='input')
  inputs = tf.io.parse_example(serialized_examples, example_spec)

  outputs = my_model(inputs, training=False)
  return tf.saved_model.predict_signature_def(inputs, outputs)

checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
hb.train.export(export_dir_base, checkpoint_path, _on_export)
```
