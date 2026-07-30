"""ctypes bindings for the compiled Mojo codec kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_IMAGEIO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-imageio.so"
)
SRC = os.path.join(ROOT, "src", "imageio.mojo")
BUILD = os.path.join(ROOT, "build", "build.sh")
I = ctypes.c_int64

_SIGNATURES = {
    "mio_png_filter": ([I, I, I, I, I, I, I, I], I),
    "mio_png_unfilter": ([I, I, I, I, I, I, I], I),
    "mio_bmp_pack": ([I, I, I, I, I, I, I, I], I),
    "mio_bmp_unpack": ([I, I, I, I, I, I, I, I, I], I),
    "mio_swap16": ([I, I, I, I, I], I),
    "mio_qoi_encode": ([I, I, I, I, I, I, I, I], I),
    "mio_qoi_decode": ([I, I, I, I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_IMAGEIO_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_IMAGEIO_LIB does not exist: {LIB}")
    stale = (
        force
        or not os.path.exists(LIB)
        or os.path.getmtime(LIB) < os.path.getmtime(SRC)
    )
    if stale:
        proc = subprocess.run(
            ["bash", BUILD], cwd=ROOT, capture_output=True, text=True, timeout=1800
        )
        if proc.returncode or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if array.nbytes == 0:
        raise ValueError("native buffers must not be empty")
    return int(array.ctypes.data)
