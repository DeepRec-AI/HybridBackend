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
