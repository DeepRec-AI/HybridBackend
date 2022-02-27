# Programming Model

HybridBackend follows a shares-nothing architecture: A HybridBackend job
consists of single-GPU workers. Workers shares nothing and coordinates by
collective communication. Each worker reads environment variable `TF_CONFIG`
for cluster information.

For nodes with more than 1 GPUs, HybridBackend provides a launcher
`hybridbackend.run` which reads environment variable `NVIDIA_VISIBLE_DEVICES` or
`CUDA_VISIBLE_DEVICES` to generate single-GPU workers.

Thanks to symmetry, the programming model of HybridBackend is simple: Use
classes and functions provided by HybridBackend to replace native counterparts,
and tune performance by configuring parameters using `hb.context.update_params`
globally or `hb.scope` / `hb.function` locally if necessary.

## Use Cases

### Launch workers on a multi-GPUs node

```bash
# Launch workers for each GPU by reading environment variable
# `NVIDIA_VISIBLE_DEVICES`.
python -m hybridbackend.run python /path/to/main.py
```

### Configure parameters globally

```python
# Hint batch size for further optimization globally.
hb.context.update_params(batch_size=64000)
```

### Configure parameters within a scope

```python
# Hint batch size for further optimization inside the scope.
with hb.scope(batch_size=64000):
  # do something.
```

### Configure parameters inside a function

```python
# Increase communicator pool size to 2 inside the function.
@hb.function(comm_pool_capacity=2)
def foo():
  # do something.
```

## APIs

```{eval-rst}
.. autoclass:: hybridbackend.tensorflow.Context
    :members:
    :special-members: __init__
.. autofunction:: hybridbackend.tensorflow.scope
.. autofunction:: hybridbackend.tensorflow.function
```
