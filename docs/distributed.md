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
hb.context.options.batch_size = 16
```

or

```bash
HB_GRAD_NBUCKETS=2 python xxx.py
```

### 1.4 Example: Launch workers on single machine of multiple GPUs

```bash
# Launch workers for each GPU by reading environment variable
# `NVIDIA_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`.
python -m hybridbackend.run python /path/to/main.py
```

### 1.5 Example: Launch workers on multiple machines of multiple GPUs

```bash
# set the environment of `TF_CONFIG` with respect to machines. E.g., 
# TF_CONFIG='{"cluster":{"chief":["x.x.x.x:8860"],"worker":["x.x.x.x:8861"]}, "task":{"type":"chief","index":0}}'
# then set `NVIDIA_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` for gpus per machine
python -m hybridbackend.run python /path/to/main.py
```

## 2. Data Parallelism

HybridBackend provides `hb.scope` to rewrite variables and optimizers for
supporting data paralleism.

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.scope
```

### 2.2 Example: Training within a scope

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

with hb.scope():
  # ...
  loss = tf.losses.get_total_loss()
  # predefined optimizer
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

## 3. Embedding-Sharded Data Parallelism

HybridBackend provides a `hb.embedding_scope` to shard variables and support
embedding-sharded data paralleism.

### 3.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.metrics.accuracy
.. autofunction:: hybridbackend.tensorflow.metrics.auc
.. autofunction:: hybridbackend.tensorflow.train.EvaluationHook
.. autofunction:: hybridbackend.tensorflow.train.export
```

### 3.2 Example: Sharding embedding weights within a scope

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

def foo():
  # ...
  with hb.scope():
    with hb.embedding_scope():
      embedding_weights = tf.get_variable(
        'emb_weights', shape=[bucket_size, dim_size])
    embedding = tf.nn.embedding_lookup(embedding_weights, ids)
    # ...
    loss = tf.losses.get_total_loss()
    # predefined optimizer
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
```

### 3.3 Example: Evaluation

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
    batch = tf.data.make_one_shot_iterator(train_ds).get_next()
    # ...
    with hb.embedding_scope():
      embedding_weights = tf.get_variable(
        'emb_weights', shape=[bucket_size, dim_size])
    embedding = tf.nn.embedding_lookup(embedding_weights, ids)
    # ...
    hooks.append(
      hb.train.EvaluationHook(
        eval_fn, every_n_iter=1000, summary_dir='/path/to/summary'))

    with tf.train.MonitoredTrainingSession('', hooks=hooks) as sess:
      while not sess.should_stop():
        sess.run(train_op)
```

### 3.4 Example: Exporting to SavedModel

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

## 4. Sync training with unbalanced data across workers.

In training data across distributed workers, it is likely that some of the 
workers have been assigned less batches of data than the others. Hence, these
workers shall run out of data ahead of other workers. HybridBackend provides users 
of two strategy to process remained training data on some of the workers. 

1. set `data_sync_drop_remainder=True` (by default) in `hb.scope()`
```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

if __name__ == '__main__':
...
with hb.scope(data_sync_drop_remainder=True):
  main()
```
By doing so, whenever one of the workers has finished assigned training data, 
HybridBackend would drop remained training data on other workers to end the 
training task.

2. set `data_sync_drop_remainder=False` in `hb.scope()`. As a result, whenever
a worker has finished its training data, it will keep producing empty data (tensor)
to join the synchronous training along with other workers until all of the workers
have finished their training data. It is worth noting that the users shall ensure 
a compatibility of their customized TF operators or other implementation to allow
such emtpy data (tensor) in their executions.
