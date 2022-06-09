# Global Operations

## 1. Embedding Layer with Sharded Weights

Recommenders in industry usually need embedding learning of large number of
categorical columns with very large embedding weights. HybridBackend supports
embedding layer with sharded weights using feature columns.

### 1.1 APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.feature_column.DenseFeatures
    :members:
    :special-members: __init__
.. autofunction:: hybridbackend.tensorflow.dense_features
```

### 1.2 Options

Option | Environment Variable | Default Value | Comment
------ | -------------------- | ------------- | --------
`emb_num_groups` | `HB_EMB_NUM_GROUPS` | `1` | Number of groups for distributed embedding lookup.
`emb_wire_dtype` | `HB_EMB_WIRE_DTYPE` | `tf.float32` | Wire data type for embedding exchange. Accepts dtype or a dict from column name to dtype.
`emb_buffer_size` | `HB_EMB_BUFFER_SIZE` | `0` | Size of embedding buffer.
`emb_backend` | - | `DEFAULT` | Backend of embeding storage.
`emb_device` | - | `''` | Device of embedding weights. Accepts string or a dict from column name to string.
`emb_dtype` | - | `tf.float32` | Data type of embedding weights. Accepts dtype or a dict from column name to dtype.
`emb_unique` | - | `False` | Whether the inputs are already unique. Accepts bool or a dict from column name to bool.
`emb_pad` | - | `False` | Whether the results should be padded. Accepts bool or a dict from column name to bool.
`emb_segment_rank` | - | `0` | Rank of the embedding to segment sum. Accepts int or a dict from column name to int.

### 1.3 Example: Embedding Layer using Feature Columns

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Specify `emb_device` to place large embedding weights to DRAM.
@hb.function(emb_device='/cpu:0')
def model_fn(features, labels, mode, params):
  # ...
  # Define embedding column: column1.
  column1 = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
          key='col1',
          num_buckets=65535,
          default_value=0),
      dimension=128)
  # Define embedding column: column2.
  column2 = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
          key='col2',
          num_buckets=1024,
          default_value=1),
      dimension=64)
  # Create a layer to lookup embeddings from column weights.
  features = hb.keras.layers.DenseFeatures([column1, column2])(batch)
  # ...
  loss = tf.losses.get_total_loss()
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  train_op = opt.minimize(loss)
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)
```

## 2. Global Metrics

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.metrics.accuracy
.. autofunction:: hybridbackend.tensorflow.metrics.auc
```

### 2.2 Example: Global AUC

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

def eval_fn():
  # ...
  auc_and_update = hb.metrics.auc(
    labels=eval_labels,
    predictions=eval_logits)
  return {'auc': auc_and_update}
```
