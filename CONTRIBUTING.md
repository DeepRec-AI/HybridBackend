# How to contribute

We appreciate all contributions to improve HybridBackend. You can create an
[issue](https://github.com/alibaba/HybridBackend/issues) or send a 
[pull request](https://github.com/alibaba/HybridBackend/pulls).

**Working on your first Pull Request?** You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://kcd.im/pull-request)

## Code style

Before any commits, please use below tools to format and check code style:

```bash
cibuild/run cibuild/format
cibuild/run cibuild/lint
```

Commit message style should follow below format:

```
[Module] Do something great.
```

`Module` could be `CI`, `IO` or other well-known abbreviations.

## Package building

Build code with default developer docker:

```bash
cibuild/run make -j8
```

## Unit tests

Test your commit in local node:

```bash
cibuild/run make test
```

Also, CI builds would be triggered if a commit is pushed.
