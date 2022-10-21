# High-Level APIs

## 1. Estimator API

Estimator is a widely used high-level API for recommendation systems.
HybridBackend provides `hb.estimator.Estimator` API for training and evaluation
using Estimator.

### 1.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.estimator.Estimator
```

### 1.2 Example: Training and Evaluation

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# ...

model = hb.estimator.Estimator(model_fn=model_fn)
model.train_and_evaluate(train_spec, eval_spec, eval_every_n_iter=1000)
```

### 1.3 Example: Exporting to SavedModel

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# ...

estimator = hb.estimator.Estimator(model_fn=model_fn)

# ...

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

### 1.4 Example: Wraps an existing estimator

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

estimator = hb.wraps(MyEstimator)(model_fn=model_fn)
```

### 1.5 Example: Customized Estimators

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.wraps
class MyEstimator(tf.estimator.Estimator):
  # ...

estimator = MyEstimator(model_fn=model_fn)
```

## 2. Feature Column API

Recommenders in industry usually need embedding learning of large number of
categorical columns with very large embedding weights. HybridBackend supports
embedding layer with sharded weights using feature columns.

### 2.1 APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.feature_column.DenseFeatures
    :members:
    :special-members: __init__
.. autofunction:: hybridbackend.tensorflow.dense_features
```

### 2.2 Options

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

### 2.3 Example: Embedding Layer using Feature Columns

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
