# Data Loading

Large batch training on cloud requires great IO performance. HybridBackend
supports memory-efficient loading of categorical data.

## 1. Data Frame

A data frame is a table consisting of multiple named columns. A named column has
a logical data type and a physical data type. Batch of values in a named column
can be read efficiently.

Supported logical data types:

| Name                        | Data Structure                                    |
| --------------------------- | ------------------------------------------------- |
| Scalar                      | `tf.Tensor` / `hb.data.DataFrame.Value`       |
| Fixed-Length List           | `tf.Tensor` / `hb.data.DataFrame.Value`       |
| Variable-Length List        | `tf.SparseTensor` / `hb.data.DataFrame.Value` |
| Variable-Length Nested List | `tf.SparseTensor` / `hb.data.DataFrame.Value` |

Supported physical data types:

| Category | Types                                                        |
| -------- | ------------------------------------------------------------ |
| Integers | `int64` `uint64` `int32` `uint32` `int8` `uint8` |
| Numerics | `float64` `float32` `float16`                          |
| Text     | `string`                                                   |

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.data.DataFrame
    :members:
    :special-members: __init__
```

## 2. Tabular Dataset

HybridBackend supports reading tabular data from
[Apache Parquet](https://parquet.apache.org) files without extra memory copy.

### 2.1 APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.data.Dataset
    :members:
    :special-members: __init__
```

### 2.3 Read and batch

Example 1:

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Read from a parquet file.
ds = hb.data.Dataset.from_parquet('/path/to/f1.parquet')
ds = ds.batch(1024)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
...
```

Example 2:

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

filenames = tf.data.Dataset.from_generator(func, tf.string, tf.TensorShape([]))
ds = hb.data.Dataset.from_parquet(filenames)
ds = ds.batch(1024)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'A': scalar_tensor, 'C': sparse_tensor}
...
```

### 2.4 Read, shuffle and batch

Example:

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

filenames = tf.data.Dataset.from_generator(func, tf.string, tf.TensorShape([]))
# Define data frame fields.
fields = [
    hb.data.DataFrame.Field('A', tf.int64),
    hb.data.DataFrame.Field('C', tf.int64, ragged_rank=1)]
# Read from parquet files by reading upstream filename dataset.
ds = hb.data.Dataset.from_parquet(filenames, fields=fields)
ds = ds.shuffle_batch(1024, buffer_size=2048)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
...
```

### 2.5 Read selected fields from files on S3/OSS/HDFS

Example:

```bash
export S3_ENDPOINT=oss-cn-shanghai-internal.aliyuncs.com
export AWS_ACCESS_KEY_ID=my_id
export AWS_SECRET_ACCESS_KEY=my_secret
export S3_ADDRESSING_STYLE=virtual
```

```{eval-rst}
.. note::
   See https://docs.w3cub.com/tensorflow~guide/deploy/s3.html for more
   information.
.. note::
   Set `S3_ADDRESSING_STYLE` to `virtual` to support OSS.
.. note::
   Set `S3_USE_HTTPS` to `0` to use `http` for S3 endpoint.
```

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Read from parquet files on remote services for selected fields.
ds = hb.data.Dataset.from_parquet(
  ['s3://path/to/f1.parquet',
   'oss://path/to/f2.parquet',
   'hdfs://host:port/path/to/f3.parquet'],
  fields=['a', 'c'])
ds = ds.batch(1024)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
...
```

### 2.6 Tensors and sparse tensors

Example:

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

# Define data frame fields.
fields = [
    hb.data.DataFrame.Field('A', tf.int64),  # scalar
    hb.data.DataFrame.Field('B', tf.int64, shape=[32]),  # fixed-length list
    hb.data.DataFrame.Field('C', tf.int64, ragged_rank=1),  # variable-length list
    hb.data.DataFrame.Field('D', tf.int64, ragged_rank=1)]  # variable-length list
# Read from parquet files by reading upstream filename dataset.
ds = hb.data.Dataset.from_parquet(
    '/path/to/f1.parquet',
    fields=fields,
    to_dense={'D': True})
ds = ds.batch(1024)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'A': scalar_tensor, 'B': list_tensor, 'C': sparse_tensor, 'D': padded_list_tensor}
...
```

## 3. Tips

### 3.1 Remove dataset ops in exported saved model

```python
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
import hybridbackend.tensorflow as hb

# ...
model_inputs = {t.name.split(':')[0]: t for t in model.inputs}
model_outputs = {t.name.split(':')[0]: t for t in model.outputs}
train_graph_def = tf.get_default_graph().as_graph_def()
predict_graph_def = TransformGraph(
  train_graph_def,
  list(model_inputs.keys()),
  list(model_outputs.keys()),
  ['strip_unused_nodes'])
with tf.Graph().as_default() as predict_graph:
  tf.import_graph_def(predict_graph_def, name='')
  with tf.Session(graph=predict_graph) as predict_sess:
    tf.saved_model.simple_save(
      predict_sess, export_dir,
      inputs=model_inputs,
      outputs=model_outputs)
```

## 4. Benchmark

In benchmark for reading 20k samples from 200 columns of a Parquet file,
`hb.data.Dataset` is about **21.51x faster** than
`TextLineDataset` + `batch` + `decode_csv` in vanilla TensorFlow using 1 CPU,
and is about **394.94x faster** using 20 CPUs. Besides, converting CSV files
into Parquet files can also reduce storage size at least **3.3x**.

| File Format      | Size (MB) | Framework     | #Threads | Step Time (ms) |
| ---------------- | --------- | ------------- | -------- | -------------- |
| CSV              | 11062.61  | TensorFlow    | 1        | 8558.38        |
| Parquet (SNAPPY) | 3346.10   | TensorFlow IO | 1        | 103056.71      |
| Parquet (SNAPPY) | 3346.10   | HybridBackend | 1        | 397.88         |
| Parquet (SNAPPY) | 3346.10   | HybridBackend | 20       | 21.67          |

```{eval-rst}
.. note::
   Above tests are taken on a machine with 96 vCPUs (Intel Xeon Platinum 8163)
   and SSD storage. Results are average of 100 iterations.
```
