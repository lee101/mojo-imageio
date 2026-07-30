"""Image encode/decode dispatch backed by Mojo kernels."""

from . import v2, v3
from ._codecs import CodecError
from .v2 import imread, imsave, imwrite, mimread, mimwrite

RETURN_BYTES = v3.RETURN_BYTES
__version__ = "0.1.0"

__all__ = [
    "CodecError",
    "RETURN_BYTES",
    "imread",
    "imsave",
    "imwrite",
    "mimread",
    "mimwrite",
    "v2",
    "v3",
]

