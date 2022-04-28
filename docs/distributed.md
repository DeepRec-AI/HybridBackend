# Training and Evaluation

## 1. Work with Estimator

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

### 1.2 Example: Exporting to SavedModel

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

### 1.3 Example: Customized Estimators

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

@hb.wraps
class MyEstimator(tf.estimator.Estimator):
  # ...

estimator = MyEstimator(model_fn=model_fn)
```

## 2. Work with MonitoredSession

HybridBackend provides low-level APIs for more precise control of training.

### 2.1 APIs

```{eval-rst}
.. autofunction:: hybridbackend.tensorflow.train.monitored_session
.. autofunction:: hybridbackend.tensorflow.data.make_one_shot_iterator
.. autofunction:: hybridbackend.tensorflow.data.make_initializable_iterator
.. autofunction:: hybridbackend.tensorflow.saved_model.export
```

### 2.2 Example: Training and Evaluation

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
    train_op = train_fn(batch)

  with hb.train.monitored_session(
      hooks=hooks,
      eval_every_n_iter=1000,
      eval_fn=eval_fn) as sess:
    while not sess.should_stop():
      sess.run(train_op)
```

### 2.3 Example: Exporting to SavedModel

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
hb.saved_model.export(export_dir_base, checkpoint_path, _on_export)
```
