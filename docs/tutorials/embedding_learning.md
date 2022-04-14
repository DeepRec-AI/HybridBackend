# Embedding Learning

Recommenders in industry usually need embedding learning of large number of
categorical columns with very large embedding weights. HybridBackend supports
parallel embedding lookup by sharding embedding weights across workers, and
utilizes weights offloading and operator coalescing to improve GPU utilization.

## Use Cases

### Traing embedding weights with fully sharding

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

with tf.device(hb.train.device_setter()):
  # ...
  # Define embedding column: column1.
  column1 = hb.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
          key='col1',
          num_buckets=65535,
          default_value=0),
      dimension=128)
  # Define embedding column: column2.
  column2 = hb.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
          key='col2',
          num_buckets=1024,
          default_value=1),
      dimension=64)
  # Define shared embedding columns.
  column34 = hb.feature_column.shared_embedding_columns(
      [tf.feature_column.categorical_column_with_identity(
           key='col3',
           num_buckets=998,
           default_value=10),
       tf.feature_column.categorical_column_with_identity(
           key='col4',
           num_buckets=356,
           default_value=8)],
      dimension=32)
  columns = [column1, column2] + column34
  # Create a layer to lookup embeddings from column weights.
  features = hb.feature_column.DenseFeatures(columns)(batch)
  # ...
  loss = tf.losses.get_total_loss()
  # Use auto-generated wrapper for predefined optimizer.
  opt = hb.train.GradientDescentOptimizer(learning_rate=lr)
  train_op = opt.minimize(loss)

# All workers must be chief to initialize variables.
with hb.train.MonitoredTrainingSession(
    server.target, is_chief=True, hooks=hooks) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```

### Train embeddings weights with cost-based sharding

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Hint `batch_size` so that different columns can use different sharding
# strategies for minimal communication cost.
# Specify `emb_device` to place large embedding weights to DRAM.
@hb.function(batch_size=64000, emb_device='/cpu:0')
def model_fn(features, labels, mode, params):
  # ...
  # Define embedding column: column1.
  column1 = hb.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
          key='col1',
          num_buckets=65535,
          default_value=0),
      dimension=128)
  # Define embedding column: column2.
  column2 = hb.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
          key='col2',
          num_buckets=1024,
          default_value=1),
      dimension=64)
  # Create a layer to lookup embeddings from column weights.
  features = hb.feature_column.DenseFeatures([column1, column2])(batch)
  # ...
  loss = tf.losses.get_total_loss()
  # Use auto-generated wrapper for predefined optimizer.
  opt = hb.train.GradientDescentOptimizer(learning_rate=lr)
  train_op = opt.minimize(loss)
  return hb.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)
# ...
# Use estimator wrapper to place model_fn automatically.
estimator = hb.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    params=params)
# ...
# Train and evaluate the estimator.
hb.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

## APIs

### Embedding Columns

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.feature_column.embedding_column
.. autofunction:: hybridbackend.tensorflow.feature_column.shared_embedding_columns
.. autoclass:: hybridbackend.tensorflow.feature_column.DenseFeatures
    :members:
    :special-members: __init__
```

## Tuning Parameters

Parameter    | Environment Variable | Default Value | Comment
------------ | -------------------- | ------------- | --------
`emb_backend` | `HB_EMB_BACKEND` | `DEFAULT` | Backend of embeding storage.
`emb_num_groups` | `HB_EMB_NUM_GROUPS` | `1` | Number of groups for communication.
`emb_device` | `HB_EMB_DEVICE` | `''` | Device of embedding weights. Accepts string or a dict from column name to string.
`emb_dtype` | `HB_EMB_DTYPE` | `tf.float32` | Data type of embedding weights. Accepts dtype or a dict from column name to dtype.
`emb_wire_dtype` | `HB_EMB_WIRE_DTYPE` | `tf.float32` | Wire data type for embedding exchange. Accepts dtype or a dict from column name to dtype.
`emb_collections` | - | `[tf.GraphKeys.GLOBAL_VARIABLES]` | Collections of embedding weights. Accepts list or a dict from column name to list.
`emb_segment_rank` | `HB_EMB_SEGMENT_RANK` | `0` | Rank of the embedding to segment sum. Accepts int or a dict from column name to int.
`emb_unique` | `HB_EMB_UNIQUE` | `False` | Whether the inputs are already unique. Accepts bool or a dict from column name to bool.
`emb_shard_unique` | `HB_EMB_SHARD_UNIQUE` | `False` | Whether the inputs for current weight shard are already unique. Accepts bool or a dict from column name to bool.
`batch_size` | `HB_BATCH_SIZE` | `None` | Hint to determine sharding weights or not
