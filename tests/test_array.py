import math

import pytest
import tensorflow as tf

import ndtensorflow as ndt


def values(x):
    return x.unwrap().numpy().tolist()


def test_array_is_owned_object_without_tensorflow_patch():
    assert not hasattr(tf.Tensor, "__array_namespace__")
    with pytest.raises(TypeError):
        ndt.Array()

    x = ndt.asarray([1, 2, 3])

    assert isinstance(x, ndt.Array)
    assert isinstance(x.unwrap(), tf.Tensor)
    assert x.__array_namespace__() is ndt
    assert not hasattr(x.unwrap(), "__array_namespace__")


def test_properties_and_dunders_return_arrays():
    x = ndt.asarray([[1, 2], [3, 4]], dtype=ndt.int32)

    assert x.shape == (2, 2)
    assert x.ndim == 2
    assert x.size == 4
    assert values(x + 1) == [[2, 3], [4, 5]]
    assert values(10 - x) == [[9, 8], [7, 6]]
    assert values(x > 2) == [[False, False], [True, True]]
    assert values(x.T) == [[1, 3], [2, 4]]


def test_indexing_and_local_assignment():
    x = ndt.asarray([1, 2, 3], dtype=ndt.int32)

    assert int(x[1]) == 2
    x[1] = 9
    assert values(x) == [1, 9, 3]

    mask = ndt.asarray([True, False, True])
    x[mask] = 0
    assert values(x) == [0, 9, 0]


def test_namespace_creation_and_manipulation():
    x = ndt.arange(6, dtype=ndt.int32)
    y = ndt.reshape(x, (2, 3))

    assert isinstance(y, ndt.Array)
    assert values(y) == [[0, 1, 2], [3, 4, 5]]
    assert values(ndt.repeat(ndt.asarray([1, 2]), 2)) == [1, 1, 2, 2]
    assert values(ndt.tensordot(y, ndt.asarray([1, 1, 1]), axes=1)) == [3, 12]


def test_linalg_returns_wrapped_results():
    x = ndt.asarray([[1.0, 2.0], [3.0, 4.0]])
    inv = ndt.linalg.solve(x, ndt.asarray([1.0, 0.0]))
    qr = ndt.linalg.qr(x)

    assert isinstance(inv, ndt.Array)
    assert inv.shape == (2,)
    assert isinstance(qr.Q, ndt.Array)
    assert isinstance(qr.R, ndt.Array)


def test_fft_returns_wrapped_results():
    x = ndt.asarray([1.0, 0.0, 0.0, 0.0])
    y = ndt.fft.rfft(x)

    assert isinstance(y, ndt.Array)
    assert y.shape == (3,)
    assert y.dtype == ndt.complex64


def test_namespace_info():
    info = ndt.__array_namespace_info__()

    assert "CPU:0" in info.devices() or info.devices()
    assert info.default_dtypes()["indexing"] == ndt.int64
    assert "float32" in info.dtypes(kind="real floating")


def test_array_api_special_value_regressions():
    z = complex(ndt.expm1(ndt.asarray(complex(math.inf, 0.0), dtype=ndt.complex128)))
    assert z.real == math.inf
    assert z.imag == 0.0
    assert math.copysign(1.0, z.imag) > 0

    z = complex(ndt.expm1(ndt.asarray(complex(-math.inf, math.nan), dtype=ndt.complex128)))
    assert z.real == -1.0
    assert z.imag == 0.0
    assert math.copysign(1.0, z.imag) > 0

    z = complex(ndt.expm1(ndt.asarray(complex(math.nan, 0.0), dtype=ndt.complex128)))
    assert math.isnan(z.real)
    assert z.imag == 0.0
    assert math.copysign(1.0, z.imag) > 0

    r = float(ndt.remainder(ndt.asarray(-0.0, dtype=ndt.float64), ndt.asarray(1.0, dtype=ndt.float64)))
    assert r == 0.0
    assert math.copysign(1.0, r) > 0

    r = float(ndt.remainder(ndt.asarray(0.0, dtype=ndt.float64), ndt.asarray(-1.0, dtype=ndt.float64)))
    assert r == 0.0
    assert math.copysign(1.0, r) < 0


def test_statistical_edge_case_regressions():
    x = ndt.asarray([], dtype=ndt.uint8)

    assert ndt.cumulative_sum(x).dtype == ndt.uint32
    assert ndt.cumulative_prod(x).dtype == ndt.uint32

    y = ndt.asarray([0.0, 0.0], dtype=ndt.float32)
    out = ndt.std(y, axis=(), correction=1)
    assert out.shape == y.shape
    assert values(out) == [0.0, 0.0]


def test_isin_uint64_scalar_regression():
    out = ndt.isin(2**63, ndt.asarray([], dtype=ndt.uint64))

    assert out.shape == ()
    assert out.dtype == ndt.bool
    assert bool(out) is False


def test_arange_float32_large_offset_regression():
    out = ndt.arange(-2147483328, -2147482646, step=21.3125, dtype=ndt.float32)

    assert out.dtype == ndt.float32
    assert out.shape == (32,)
