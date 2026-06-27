# ndtensorflow

`ndtensorflow` is an experimental TensorFlow-backed Array API namespace.

The important design choice is that the public array object is owned by this
project: `ndtensorflow.Array` wraps a `tf.Tensor`. TensorFlow classes are not
patched, and TensorFlow tensors are not asked to provide Array API object
methods.

This follows the same high-level shape as `ndonnx`: arrays are user-facing
wrapper objects, `__array_namespace__()` returns the package namespace, and
operations unwrap backend values, call the backend, then wrap results again.
