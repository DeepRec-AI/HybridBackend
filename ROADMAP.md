# HybridBackend Roadmap

## HybridBackend v0.6 (2022-05)

Objective: "Communication-efficient training and evaluation at scale"

- Data-Parallel Training and Evaluation
  - Bucketized Gradients Aggregation using AllReduce
  - Global Metric Operations
  - Out-Of-Range Coordination

- Hybrid-Parallel Embedding Learning
  - Bucketized Embedding Exchanging using AllToAllv
  - Fusion and Quantization of AllToAllv
  - Fusion of Partitioning and Stitching

Objective: "Easy to use with existing AI workflows"

- Usability
  - Support of MonitoredSession and Estimator
  - Declarative API for Model Definition

- Compatibility
  - Support of NVIDIA TensorFlow and DeepRec

- Interoperability
  - Inference Pipeline Needs No Change
  - Support of SavedModel
  - Support of Variable, XDL HashTable and PAI Embedding Variable

## HybridBackend v0.5 (2021-11)

Objective: "Memory-efficient loading of categorical data"

- Parquet Dataset
  - Reading batch of tensors from numeric fields in zero-copy way
  - Reading batch of sparse tensors from numeric list fields in zero-copy way
  - Support of string fields
  - Support of local filesystem, HDFS, S3 and OSS

- Data Pipeline Functions
  - Resizing batch of tensors and ragged tensors
  - Converting ragged tensors to sparse tensors

Objective: "Easy to use with existing AI workflows"

- Compatibility
  - Support of TensorFlow 1.15 and Tensorflow 1.14
  - GitHub actions for uploading wheels to PyPI
