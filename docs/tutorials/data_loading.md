# Data Loading

Distributed training on cloud requires great IO performance. HybridBackend
supports direct reading of batches from
[Apache Parquet](https://parquet.apache.org) files without extra memory copy.

## Data Frame

A data frame is a table consisting of multiple named columns. A named column has
a logical data type and a physical data type. Batch of values in a named column
can be read efficiently.

Supported logical data types:

Name        | Data Structure
----------- | ------------------------
Scalar      | `tf.Tensor`
Nested List | `hb.data.DataFrame.Value` = `values` + `nested_row_splits`

Supported physical data types:

Category | Types
-------- | ---------
Integers | `int64` `uint64` `int32` `uint32` `int8` `uint8`
Numerics | `float64` `float32` `float16`
Text     | `string`

## Data Loading APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.data.ParquetDataset
    :members:
    :special-members: __init__
.. autoclass:: hybridbackend.tensorflow.data.DataFrame
    :members:
    :special-members: __init__
.. autofunction:: hybridbackend.tensorflow.data.read_parquet
.. autofunction:: hybridbackend.tensorflow.data.to_sparse
```

## Data Loading Use Cases

### Read from one file

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb
ds = hb.data.ParquetDataset(
    '/path/to/f1.parquet',
    batch_size=1024)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
...
```

### Read sparse tensors from selected fields

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb
ds = hb.data.ParquetDataset(
    ['/path/to/f1.parquet', '/path/to/f2.parquet'],
    batch_size=1024,
    fields=['a', 'c'])
ds = ds.apply(hb.data.to_sparse())
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
...
```

### Read from filenames dataset

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb
filenames = tf.data.Dataset.from_generator(func, tf.string, tf.TensorShape([]))
fields = [
    hb.data.DataFrame.Field('A', tf.int64),
    hb.data.DataFrame.Field('C', tf.int64, ragged_rank=1)]
ds = filenames.apply(hb.data.read_parquet(1024, fields=fields))
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
...
```

## Performance

In benchmark for reading 20k samples from 200 columns of a Parquet file,
`hb.data.ParquetDataset` is about **21.51x faster** than
`TextLineDataset` + `batch` + `decode_csv` in vanilla TensorFlow using 1 CPU,
and is about **394.94x faster** using 20 CPUs. Besides, converting CSV files
into Parquet files can also reduce storage size at least **3.3x**.

File Format | Size (MB) | Framework     | #Threads | Elapsed (ms)
----------- | --------- | ------------- | -------- | ------------
CSV         | 11062.61  | Tensorflow    | 1        | 8558.38
Parquet     | 3346.10   | Tensorflow IO | 1        | 103056.71
Parquet     | 3346.10   | HybridBackend | 1        | 397.88
Parquet     | 3346.10   | HybridBackend | 20       | 21.67

```{eval-rst}
.. note::
   Set `MALLOC_CONF` to `"background_thread:true,metadata_thp:auto"` to speed
   up memory access.
.. note::
   Set `ARROW_NUM_THREADS` to read different columns in parallel.
```
