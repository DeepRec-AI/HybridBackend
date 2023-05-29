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

### 1.6 Use DeepRec Embedding Variable by Estimator

In Estimator API, we must use `tf.feature_column` API to enable the usage of
DeepRec's [Embedding Variable](https://deeprec.readthedocs.io/en/latest/Embedding-Variable.html) functionality
Besides, it requires a `variable_scope` with a fixed partitioner upon
the `input_layer` to enable sharding embedding variables across workers.

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

def mode_fn(features, labels, mode, params):
    ...
    embedding_columns = [
      tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_embedding(
          fs.name, dtype=tf.as_dtype(fs.dtype)),
        dimension=self._args.data_spec.embedding_dims[fs.name],
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
      for fs in categorical_fields]
    ...
    with hb.embedding_scope(), tf.device('/cpu:0'):
      with tf.variable_scope(
        'embedding',
          partitioner=tf.fixed_size_partitioner(hb.context.world_size)):
        deep_features = [
          tf.feature_column.input_layer(features, [c])
          for c in embedding_columns]
```

## 2. Keras API (experimental)

`tf.keras` is also a widespred tensorflow API. However, the `tf.keras` API 
of `tf.1.x` has an incomplete support for distributed training. For example, 
it does not support a model-parallel training strategy requested by recommender 
systems. HybridBackend provides a `hb.keras` API for training and evaluation in
both of data-parallel and model-parallel strategies.

### 2.1 APIs 

`hb.keras` provides additional arguments in `compile` and `fit` methods
when compared to `tf.keras`

1. `compile` method

```python
def compile(self,
            ...
            clipnorm=None,
            clipvalue=None,
            **kwargs):
```
, where `clipnorm` and `clipvalue` are set to accomplish gradient clipping. 

2. `fit` method

```python
def fit(self,
        ...
        checkpoint_dir=None,
        keep_checkpoint_max=None,
        keep_checkpoint_every_n_hours=None,
        monitor=`val_loss`,
        save_best_only=False,
        mode=`auto`
        ...
        ):
```
, where `checkpoint_dir` accepts paths to store and restore checkpoint files.
Currently, it only supports producing a format of `tf.train.Checkpoint`.
Besides, users can specify the frequency of saving checkpoint and how to 
save checkpoints with a best monitored value.

### 2.2 Example: Using Keras's functional API

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# ...

train_dataset = input_dataset(
  args, train_filenames, args.train_batch_size)
val_dataset = input_dataset(
  args, val_filenames, args.eval_batch_size)

features, labels = tf.data.make_one_shot_iterator(train_dataset).get_next()
model_output = RankingModel(args)(features)

dcnv2_in_keras = tf.keras.Model(inputs=[features], outputs=model_output)
dcnv2_in_keras.compile(
  loss=loss_func,
  metrics=[tf.keras.metrics.AUC()],
  optimizer=opt,
  target_tensors=labels,
  clipnorm=1.0,
  clipvalue=1.0)

dcnv2_in_keras.fit(
  x=None,
  y=None,
  epochs=1,
  validation_data=val_dataset,
  batch_size=args.train_batch_size,
  validation_steps=args.eval_max_steps,
  steps_per_epoch=args.train_max_steps,
  checkpoint_dir=args.output_dir,
  keep_checkpoint_max=2,
  monitor='val_auc',
  mode='max',
  save_best_only=True)

dcnv2_in_keras.export_saved_model(
  args.output_dir,
  lambda: predict_fn(args))
```

### 2.3 Example: Using Keras's Subclassing API

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# ...

class DlrmInKeras(hb.keras.Model):
  def __init__(self, args):
    super().__init__()
    self._args = args
    self.dlrm = RankingModel(args)

  def call(self, inputs):  # pylint: disable=method-hidden
    return self.dlrm(inputs)

...

dlrm_in_keras = DlrmInKeras(args)
opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
def loss_func(y_true, y_pred):
  return tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(y_true, y_pred))

dlrm_in_keras.compile(
  loss=loss_func,
  metrics=[tf.keras.metrics.AUC()],
  optimizer=opt)

dlrm_in_keras.fit(
  x=train_dataset,
  y=None,
  epochs=1,
  validation_data=val_dataset,
  batch_size=args.train_batch_size,
  validation_steps=args.eval_max_steps,
  steps_per_epoch=args.train_max_steps,
  checkpoint_dir=args.output_dir,
  keep_checkpoint_max=2,
  monitor='val_auc',
  mode='max',
  save_best_only=True)

dlrm_in_keras.summary()
dlrm_in_keras.export_saved_model(
  args.output_dir,
  lambda: predict_fn(args))
```
Currently, `hb.keras` only supports optimizers from
`tf.compat.v1.train.Optimizer` rather than `tf.keras.optimizers`

### 2.4 Example: load a checkpoint to initialize parameters.

HB provides a API to load a pre-trained checkpoint file for both of functional
API and subclassing API to initialize parameters.

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

...

dlrm_in_keras = DlrmInKeras(args)
dlrm_in_keras.load_weights(args.weights_dir)
```
It is worth noting that the checkpoint file in weights directory must be in 
format of `tf.train.Checkpoint` rather than the 
format of `tf.keras.callbacks.ModelCheckpoint` (currently not support)
