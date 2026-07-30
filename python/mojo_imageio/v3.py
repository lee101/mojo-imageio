from __future__ import annotations

import os
from typing import BinaryIO

import numpy as np

from ._codecs import (
    CodecError,
    decode_bmp,
    decode_png,
    decode_pnm,
    decode_qoi,
    encode_bmp,
    encode_png,
    encode_pnm,
    encode_qoi,
)

RETURN_BYTES = "<bytes>"

_ALIASES = {
    "png": "png",
    "bmp": "bmp",
    "dib": "bmp",
    "pbm": "pbm",
    "pgm": "pgm",
    "ppm": "ppm",
    "pnm": "pnm",
    "qoi": "qoi",
}


def _extension(value) -> str | None:
    if value is None:
        return None
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    text = text.lower().rsplit("/", 1)[-1]
    suffix = text.rsplit(".", 1)[-1] if "." in text else text
    return _ALIASES.get(suffix)


def _read(uri) -> tuple[bytes, str | None]:
    if isinstance(uri, (bytes, bytearray, memoryview)):
        return bytes(uri), None
    if hasattr(uri, "read"):
        data = uri.read()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("image file object must return bytes")
        return bytes(data), _extension(getattr(uri, "name", None))
    path = os.fspath(uri)
    with open(path, "rb") as stream:
        return stream.read(), _extension(path)


def _detect(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"qoif"):
        return "qoi"
    if len(data) >= 2 and data[:2] in (
        b"P1",
        b"P2",
        b"P3",
        b"P4",
        b"P5",
        b"P6",
    ):
        return "pnm"
    raise CodecError("no supported image codec recognizes this data")


def _requested_format(
    uri, extension=None, format_hint=None, plugin=None
) -> str | None:
    for value in (extension, format_hint):
        selected = _extension(value)
        if selected:
            return selected
    if plugin:
        selected = _extension(plugin)
        if selected:
            return selected
    if not isinstance(uri, (bytes, bytearray, memoryview)) and not hasattr(
        uri, "write"
    ):
        return _extension(uri)
    return _extension(getattr(uri, "name", None))


def imread(
    uri,
    *,
    index=None,
    plugin=None,
    extension=None,
    format_hint=None,
    **kwargs,
) -> np.ndarray:
    if index not in (None, 0, Ellipsis):
        raise IndexError("covered codecs contain exactly one image")
    data, _ = _read(uri)
    detected = _detect(data)
    hint = _requested_format(uri, extension, format_hint, plugin)
    if hint and hint != "pnm":
        hint_family = "pnm" if hint in ("pbm", "pgm", "ppm") else hint
        if hint_family != detected and format_hint is None:
            raise CodecError(
                f"requested {hint_family.upper()} but data is {detected.upper()}"
            )
    decoder = {
        "png": decode_png,
        "bmp": decode_bmp,
        "pnm": decode_pnm,
        "qoi": decode_qoi,
    }[detected]
    return decoder(data, **kwargs)


def imwrite(
    uri,
    image,
    *,
    plugin=None,
    extension=None,
    format_hint=None,
    **kwargs,
):
    selected = _requested_format(uri, extension, format_hint, plugin)
    if selected is None:
        raise ValueError("could not infer output format; pass extension='.png', etc.")
    if selected == "png":
        data = encode_png(image, **kwargs)
    elif selected == "bmp":
        data = encode_bmp(image, **kwargs)
    elif selected in ("pbm", "pgm", "ppm", "pnm"):
        data = encode_pnm(image, selected, **kwargs)
    elif selected == "qoi":
        data = encode_qoi(image, **kwargs)
    else:
        raise ValueError(f"unsupported output format {selected!r}")
    if uri == RETURN_BYTES:
        return data
    if hasattr(uri, "write"):
        written = uri.write(data)
        if written is not None and written != len(data):
            raise OSError(f"short image write: wrote {written} of {len(data)} bytes")
        return None
    path = os.fspath(uri)
    with open(path, "wb") as stream:
        written = stream.write(data)
        if written != len(data):
            raise OSError(f"short image write: wrote {written} of {len(data)} bytes")
    return None


def imiter(
    uri,
    *,
    index=None,
    plugin=None,
    extension=None,
    format_hint=None,
    **kwargs,
):
    yield imread(
        uri,
        index=index,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **kwargs,
    )
