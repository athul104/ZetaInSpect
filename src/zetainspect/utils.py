# utils.py
from __future__ import annotations
import numpy as np

__all__ = [
    # array/shape
    "asarray_numeric",
    "as_batch",
    "as_grid_1d",
    "broadcast_batch_grid",

    # numeric safety
    "isclose_zero",
    "safe_divide",
    "clamp_positive",

    # validation helpers
    "require_strictly_decreasing",
    "require_strictly_increasing",
]

# -------------------------
# Array + shape helpers
# -------------------------

def asarray_numeric(x, *, dtype=np.float64) -> np.ndarray:
    """
    np.asarray with dtype enforcement and object-dtype guard.
    Use this everywhere you convert user inputs.
    """
    arr = np.asarray(x, dtype=dtype)
    if arr.dtype == object:
        raise TypeError("Object dtype detected; pass numeric arrays/lists.")
    return arr


def as_batch(x, *, dtype=np.float64) -> np.ndarray:
    """
    Ensure a batch axis exists.
    Scalar -> (1,)
    (B, ...) unchanged
    """
    arr = asarray_numeric(x, dtype=dtype)
    if arr.ndim == 0:
        arr = arr[None]
    return arr


def as_grid_1d(x, *, dtype=np.float64) -> np.ndarray:
    """
    Ensure x is a 1D grid array of shape (T,).
    """
    arr = asarray_numeric(x, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D grid of shape (T,), got {arr.shape}")
    return arr


def broadcast_batch_grid(param_batch, grid_1d, *, dtype=np.float64):
    """
    Common broadcasting pattern:
      param_batch: (B, ...)  -> (B, ..., 1)
      grid_1d:     (T,)      -> (1, T)

    Returns (param_view, grid_view) as views that broadcast in elementwise ops.
    """
    p = as_batch(param_batch, dtype=dtype)
    g = as_grid_1d(grid_1d, dtype=dtype)
    return p[..., None], g[None, :]


# -------------------------
# Numerical safety
# -------------------------

def isclose_zero(x, *, atol=1e-12, rtol=1e-12) -> np.ndarray:
    """Vectorized ~0 test for floats."""
    x = np.asarray(x)
    return np.isclose(x, 0.0, atol=atol, rtol=rtol)


def safe_divide(num, den, *, default=0.0, atol=1e-300, rtol=0.0):
    """
    Elementwise num/den with protection against ~0 denominator.
    Returns an array; never raises ZeroDivisionError.
    """
    num = np.asarray(num)
    den = np.asarray(den)

    out_shape = np.broadcast(num, den).shape
    out_dtype = np.result_type(num, den, np.asarray(default))
    out = np.full(out_shape, default, dtype=out_dtype)

    good = ~np.isclose(den, 0.0, atol=atol, rtol=rtol)
    np.divide(num, den, out=out, where=good)
    return out


def clamp_positive(x, *, eps=1e-300):
    """
    Clamp x to >= eps (useful before log/division).
    """
    x = np.asarray(x)
    return np.maximum(x, eps)


# -------------------------
# Validation helpers (small, but used everywhere)
# -------------------------

def require_strictly_decreasing(x, *, name="array", tol=0.0):
    """
    Raise if x is not strictly decreasing.
    tol allows tiny floating jitter (e.g. tol=1e-14).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    d = np.diff(x)
    if not np.all(d < -tol):
        raise ValueError(f"{name} must be strictly decreasing.")


def require_strictly_increasing(x, *, name="array", tol=0.0):
    """
    Raise if x is not strictly increasing.
    tol allows tiny floating jitter (e.g. tol=1e-14).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    d = np.diff(x)
    if not np.all(d > tol):
        raise ValueError(f"{name} must be strictly increasing.")
