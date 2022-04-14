# Parallel Training

Synchronous distributed training of deep recommenders on GPU using parameter
servers usually suffers from poor network performance. HybridBackend introduces
collective communication based parallel training to speed up training.

## Use Cases

### Use a predefined optimizer

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# ...
loss = tf.losses.get_total_loss()
opt = hb.train.GradientDescentOptimizer(learning_rate=lr)
```

### Use a customized optimizer

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.train.wraps_optimizer
class MyOptimizer(tf.train.Optimizer):
  # ...

# ...
loss = tf.losses.get_total_loss()
opt = MyOptimizer(learning_rate=lr)
train_op = opt.minimize(loss)
```

### Work with MonitoredTrainingSession

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Place graph on specific GPU.
with tf.device(hb.train.device_setter()):
  # Read from a parquet file.
  with tf.device('/cpu:0'):
    train_ds = hb.data.ParquetDataset(
        '/path/to/f1.parquet',
        batch_size=1024)
    train_ds = train_ds.prefetch(4)
    eval_ds = hb.data.ParquetDataset(
        '/path/to/f2.parquet',
        batch_size=1024)
    eval_ds = eval_ds.prefetch(4)
  # ...
  train_it = hb.data.make_iterator(train_ds)
  train_features = train_it.get_next()
  # {'a': tensora, 'c': tensorc}
  # ...
  loss = tf.losses.get_total_loss()
  # Use auto-generated wrapper for predefined optimizer.
  opt = hb.train.GradientDescentOptimizer(learning_rate=lr)
  train_op = opt.minimize(loss)

  # Update evaluation metrics
  eval_update_op = ...
  eval_metric_dict = ...

# Create a server by reading cluster information in `hb.context`.
server = hb.train.Server()

# All workers must be chief to initialize variables.
with hb.train.MonitoredTrainingSession(
    server.target,
    is_chief=True,
    hooks=[...]) as sess:
  while not sess.should_stop():
    sess.run(train_op)

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
hb.saved_model.export(export_dir_base, checkpoint_path, _on_export)
```

### Work with estimator

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

def model_fn(features, labels, mode, params):
  # ...
  if mode == tf.estimator.ModeKeys.TRAIN:
    opt = hb.train.AdagradOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)
    step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=step)
    return hb.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    # Parallel evaluation is enabled by default.
    # ...
  if mode == tf.estimator.ModeKeys.PREDICT:
    # ...

model = hb.estimator.Estimator(model_fn=model_fn)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

# Train and evaluate the estimator.
hb.estimator.train_and_evaluate(model, train_spec, eval_spec)

def _on_export():
  inputs = {}
  for f in numeric_fields:
    inputs[f] = tf.placeholder(dtype=tf.int64, shape=[None], name=f)
  for f in categorical_fields:
    # Feed sparse placeholders using `tf.SparseTensorValue`
    inputs[f] = tf.sparse_placeholder(dtype=tf.int64, name=f)
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

estimator.export_saved_model(export_dir_base, _on_export)
```

## APIs

### Optimizer Wrapper

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.train.wraps_optimizer
```

```{eval-rst}
.. note::
   Every predefined optimizer, e.g. `tf.train.AdamOptimizer`, has an
   auto-generated wrapper, e.g. `hb.train.AdamOptimizer`.
```

### Server and Session

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.train.Server
.. autofunction:: hybridbackend.tensorflow.train.MonitoredTrainingSession
.. autofunction:: hybridbackend.tensorflow.train.device_setter
```

### Estimator

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.estimator.Estimator
.. autofunction:: hybridbackend.tensorflow.estimator.train_and_evaluate
```

### Dataset Iterator

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.data.make_iterator
```

### Saved Model

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.saved_model.export
```

## Tuning Parameters

Parameter    | Environment Variable | Default Value | Comment
------------ | -------------------- | ------------- | --------
`comm_default` | `HB_COMM_DEFAULT` | `NCCL` | Implementation of communicators
`comm_pool_capacity` | `HB_COMM_POOL_CAPACITY` | `1` | Number of communicators for communication
