from __future__ import annotations

import binascii
import struct

import numpy as np
from zlib_ng import zlib_ng as zlib

from ._lib import addr, lib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
QOI_PADDING = b"\x00\x00\x00\x00\x00\x00\x00\x01"


class CodecError(ValueError):
    pass


def _u8_buffer(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(
        ">I", binascii.crc32(body) & 0xFFFFFFFF
    )


def encode_png(image, **kwargs) -> bytes:
    compress_level = kwargs.pop(
        "compress_level", kwargs.pop("compression_level", 6)
    )
    filter_type = kwargs.pop("filter_type", -1)
    kwargs.pop("optimize", None)
    if kwargs:
        raise TypeError(f"unsupported PNG option: {next(iter(kwargs))}")
    if not isinstance(compress_level, int) or not 0 <= compress_level <= 9:
        raise ValueError("compress_level must be an integer from 0 through 9")
    if filter_type not in (-1, 0, 1, 2, 3, 4):
        raise ValueError("filter_type must be -1 or an integer from 0 through 4")

    array = np.asarray(image)
    if array.ndim == 2:
        height, width = array.shape
        channels = 1
    elif array.ndim == 3 and array.shape[2] in (2, 3, 4):
        height, width, channels = array.shape
    else:
        raise ValueError("PNG expects HxW, HxWx2, HxWx3, or HxWx4 image data")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")

    if array.dtype == np.uint8:
        bit_depth = 8
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
    elif array.dtype == np.uint16 and channels == 1:
        bit_depth = 16
        raw = np.ascontiguousarray(array.astype(">u2", copy=False)).view(
            np.uint8
        ).reshape(-1)
    else:
        raise TypeError("PNG supports uint8 images and uint16 grayscale images")

    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    bytes_per_sample = bit_depth // 8
    stride = width * channels * bytes_per_sample
    filtered = np.empty(height * (stride + 1), dtype=np.uint8)
    status = lib().mio_png_filter(
        addr(raw),
        raw.nbytes,
        addr(filtered),
        filtered.nbytes,
        height,
        stride,
        channels * bytes_per_sample,
        filter_type,
    )
    if status:
        raise CodecError("native PNG filter rejected buffer metadata")
    ihdr = struct.pack(
        ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0
    )
    compressed = zlib.compress(filtered, level=compress_level)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _read_png_chunks(data: bytes) -> tuple[dict[bytes, list[bytes]], bytes]:
    if not data.startswith(PNG_SIGNATURE):
        raise CodecError("not a PNG image")
    chunks: dict[bytes, list[bytes]] = {}
    pos = len(PNG_SIGNATURE)
    while pos + 12 <= len(data):
        size = struct.unpack_from(">I", data, pos)[0]
        end = pos + 12 + size
        if end > len(data):
            raise CodecError("truncated PNG chunk")
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + size]
        expected = struct.unpack_from(">I", data, pos + 8 + size)[0]
        actual = binascii.crc32(kind + payload) & 0xFFFFFFFF
        if actual != expected:
            raise CodecError(f"bad PNG CRC in {kind.decode('ascii', 'replace')}")
        chunks.setdefault(kind, []).append(payload)
        pos = end
        if kind == b"IEND":
            return chunks, data[pos:]
    raise CodecError("PNG has no complete IEND chunk")


def decode_png(data: bytes, **kwargs) -> np.ndarray:
    mode = kwargs.pop("mode", None)
    kwargs.pop("pilmode", None)
    if kwargs:
        raise TypeError(f"unsupported PNG option: {next(iter(kwargs))}")
    chunks, _ = _read_png_chunks(data)
    if b"IHDR" not in chunks or b"IDAT" not in chunks:
        raise CodecError("PNG is missing IHDR or IDAT")
    if len(chunks[b"IHDR"]) != 1 or len(chunks[b"IHDR"][0]) != 13:
        raise CodecError("invalid PNG IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", chunks[b"IHDR"][0])
    )
    if width == 0 or height == 0:
        raise CodecError("invalid zero-sized PNG")
    if compression != 0 or filtering != 0:
        raise CodecError("unsupported PNG compression or filter method")
    if interlace != 0:
        raise CodecError("interlaced PNG is not supported")
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise CodecError(f"unsupported PNG color type {color_type}")
    if bit_depth not in (8, 16) or (bit_depth == 16 and color_type == 3):
        raise CodecError(
            "only 8-bit PNG and non-palette 16-bit PNG are supported"
        )
    channels = channels_by_type[color_type]
    bytes_per_sample = bit_depth // 8
    stride = width * channels * bytes_per_sample
    expected_size = height * (stride + 1)
    try:
        inflater = zlib.decompressobj()
        inflated = inflater.decompress(b"".join(chunks[b"IDAT"]), expected_size + 1)
    except zlib.error as exc:
        raise CodecError(f"invalid PNG deflate stream: {exc}") from exc
    if (
        len(inflated) != expected_size
        or not inflater.eof
        or inflater.unconsumed_tail
        or inflater.unused_data
    ):
        raise CodecError(
            f"PNG scanline data has {len(inflated)} bytes, expected {expected_size}"
        )
    source = _u8_buffer(inflated)
    raw = np.empty(height * stride, dtype=np.uint8)
    if lib().mio_png_unfilter(
        addr(source),
        source.nbytes,
        addr(raw),
        raw.nbytes,
        height,
        stride,
        channels * bytes_per_sample,
    ):
        raise CodecError("invalid PNG scanline filter")

    if bit_depth == 16:
        array = raw.view(">u2").astype(np.uint16).reshape(
            (height, width) if channels == 1 else (height, width, channels)
        )
    else:
        array = raw.reshape(
            (height, width) if channels == 1 else (height, width, channels)
        )
        if color_type == 3:
            if b"PLTE" not in chunks:
                raise CodecError("palette PNG has no PLTE chunk")
            palette_bytes = chunks[b"PLTE"][0]
            if len(palette_bytes) % 3 or not 0 < len(palette_bytes) <= 768:
                raise CodecError("invalid PNG palette")
            palette = np.frombuffer(palette_bytes, dtype=np.uint8).reshape(-1, 3)
            if int(array.max(initial=0)) >= len(palette):
                raise CodecError("PNG palette index is out of range")
            array = palette[array]
            if b"tRNS" in chunks and mode is not None and mode.upper() == "RGBA":
                alpha = np.full(len(palette), 255, dtype=np.uint8)
                transparency = np.frombuffer(chunks[b"tRNS"][0], dtype=np.uint8)
                if len(transparency) > len(palette):
                    raise CodecError("PNG transparency table exceeds its palette")
                alpha[: len(transparency)] = transparency
                array = np.concatenate((array, alpha[raw.reshape(height, width), None]), axis=2)
    return _convert_mode(array, mode)


def _convert_mode(array: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return array
    normalized = mode.upper()
    if normalized in ("L", "I") and array.ndim == 2:
        return array
    if normalized == "RGB":
        if array.ndim == 2:
            return np.repeat(array[:, :, None], 3, axis=2)
        if array.shape[2] == 2:
            return np.repeat(array[:, :, :1], 3, axis=2)
        return array[:, :, :3]
    if normalized == "RGBA":
        if array.ndim == 2:
            rgb = np.repeat(array[:, :, None], 3, axis=2)
            alpha = np.full(
                (*array.shape, 1), np.iinfo(array.dtype).max, dtype=array.dtype
            )
            return np.concatenate((rgb, alpha), axis=2)
        if array.shape[2] == 2:
            rgb = np.repeat(array[:, :, :1], 3, axis=2)
            return np.concatenate((rgb, array[:, :, 1:2]), axis=2)
        if array.shape[2] == 4:
            return array
        alpha = np.full(
            (*array.shape[:2], 1), np.iinfo(array.dtype).max, dtype=array.dtype
        )
        return np.concatenate((array[:, :, :3], alpha), axis=2)
    raise ValueError(f"unsupported output mode {mode!r}")


def encode_bmp(image, **kwargs) -> bytes:
    if kwargs:
        raise TypeError(f"unsupported BMP option: {next(iter(kwargs))}")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("BMP supports uint8 images")
    if array.ndim == 2:
        height, width = array.shape
        channels = 1
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, channels = array.shape
    else:
        raise ValueError("BMP expects HxW grayscale or HxWx3 RGB data")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    source = np.ascontiguousarray(array).reshape(-1)
    row_stride = (width * channels + 3) & ~3
    pixels = np.empty(height * row_stride, dtype=np.uint8)
    if lib().mio_bmp_pack(
        addr(source),
        source.nbytes,
        addr(pixels),
        pixels.nbytes,
        width,
        height,
        channels,
        row_stride,
    ):
        raise CodecError("native BMP pack rejected buffer metadata")
    palette = b""
    if channels == 1:
        palette = b"".join(bytes((i, i, i, 0)) for i in range(256))
    offset = 14 + 40 + len(palette)
    file_size = offset + pixels.nbytes
    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, offset)
    dib = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        channels * 8,
        0,
        pixels.nbytes,
        0,
        0,
        256 if channels == 1 else 0,
        0,
    )
    return file_header + dib + palette + pixels.tobytes()


def decode_bmp(data: bytes, **kwargs) -> np.ndarray:
    mode = kwargs.pop("mode", None)
    if kwargs:
        raise TypeError(f"unsupported BMP option: {next(iter(kwargs))}")
    if len(data) < 54 or data[:2] != b"BM":
        raise CodecError("not a complete BMP image")
    _, declared_size, _, _, pixel_offset = struct.unpack_from("<2sIHHI", data)
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40 or len(data) < 14 + dib_size:
        raise CodecError("unsupported or truncated BMP header")
    (
        _,
        width,
        signed_height,
        planes,
        bits,
        compression,
        image_size,
        _,
        _,
        colors_used,
        _,
    ) = struct.unpack_from("<IiiHHIIiiII", data, 14)
    if width <= 0 or signed_height == 0 or planes != 1:
        raise CodecError("invalid BMP dimensions or plane count")
    if compression != 0 or bits not in (8, 24, 32):
        raise CodecError("only uncompressed 8-bit, 24-bit, and 32-bit BMP is supported")
    height = abs(signed_height)
    channels = bits // 8
    row_stride = (width * channels + 3) & ~3
    needed = pixel_offset + height * row_stride
    if pixel_offset < 14 + dib_size:
        raise CodecError("BMP pixel data overlaps its header")
    if needed > len(data):
        raise CodecError("truncated BMP pixel data")
    if declared_size and declared_size > len(data):
        raise CodecError("truncated BMP file")
    source = _u8_buffer(data[pixel_offset:needed])
    raw = np.empty(height * width * channels, dtype=np.uint8)
    if lib().mio_bmp_unpack(
        addr(source),
        source.nbytes,
        addr(raw),
        raw.nbytes,
        width,
        height,
        channels,
        row_stride,
        int(signed_height < 0),
    ):
        raise CodecError("native BMP unpack rejected buffer metadata")
    if channels == 1:
        indices = raw.reshape(height, width)
        palette_count = colors_used or 256
        palette_start = 14 + dib_size
        palette_end = palette_start + palette_count * 4
        if palette_count > 256 or palette_end > pixel_offset:
            raise CodecError("invalid BMP palette")
        palette_bgra = np.frombuffer(
            data[palette_start:palette_end], dtype=np.uint8
        ).reshape(-1, 4)
        palette = palette_bgra[:, [2, 1, 0]]
        identity = (
            len(palette) == 256
            and np.array_equal(palette[:, 0], np.arange(256, dtype=np.uint8))
            and np.array_equal(palette[:, 0], palette[:, 1])
            and np.array_equal(palette[:, 0], palette[:, 2])
        )
        if int(indices.max(initial=0)) >= len(palette):
            raise CodecError("BMP palette index is out of range")
        array = indices if identity else palette[indices]
    else:
        packed = raw.reshape(height, width, channels)
        array = packed if channels == 3 else packed[:, :, :3].copy()
    return _convert_mode(array, mode)


def _pnm_token(data: bytes, pos: int) -> tuple[bytes, int]:
    size = len(data)
    while pos < size:
        if data[pos] in b" \t\r\n\v\f":
            pos += 1
        elif data[pos] == 35:
            newline = data.find(b"\n", pos + 1)
            pos = size if newline < 0 else newline + 1
        else:
            break
    start = pos
    while pos < size and data[pos] not in b" \t\r\n\v\f#":
        pos += 1
    if start == pos:
        raise CodecError("truncated PNM header")
    return data[start:pos], pos


def _pnm_binary_start(data: bytes, pos: int) -> int:
    if pos >= len(data) or data[pos] not in b" \t\r\n\v\f":
        raise CodecError("PNM header has no payload separator")
    if data[pos : pos + 2] == b"\r\n":
        return pos + 2
    return pos + 1


def decode_pnm(data: bytes, **kwargs) -> np.ndarray:
    if kwargs:
        raise TypeError(f"unsupported PNM option: {next(iter(kwargs))}")
    magic, pos = _pnm_token(data, 0)
    if magic not in (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"):
        raise CodecError("not a supported PNM image")
    width_token, pos = _pnm_token(data, pos)
    height_token, pos = _pnm_token(data, pos)
    try:
        width, height = int(width_token), int(height_token)
    except ValueError as exc:
        raise CodecError("invalid PNM dimensions") from exc
    if width <= 0 or height <= 0:
        raise CodecError("invalid PNM dimensions")
    channels = 3 if magic in (b"P3", b"P6") else 1
    count = width * height * channels

    if magic in (b"P1", b"P4"):
        maxval = 1
    else:
        max_token, pos = _pnm_token(data, pos)
        try:
            maxval = int(max_token)
        except ValueError as exc:
            raise CodecError("invalid PNM maxval") from exc
        if not 0 < maxval <= 65535:
            raise CodecError("PNM maxval must be from 1 through 65535")

    if magic in (b"P1", b"P2", b"P3"):
        values = []
        for _ in range(count):
            token, pos = _pnm_token(data, pos)
            try:
                values.append(int(token))
            except ValueError as exc:
                raise CodecError("invalid ASCII PNM sample") from exc
        if any(value < 0 or value > maxval for value in values):
            raise CodecError("PNM sample is outside maxval")
        dtype = np.uint8 if maxval < 256 else np.uint16
        array = np.asarray(values, dtype=dtype)
    else:
        payload_start = _pnm_binary_start(data, pos)
        payload = data[payload_start:]
        if magic == b"P4":
            row_bytes = (width + 7) // 8
            needed = height * row_bytes
            if len(payload) < needed:
                raise CodecError("truncated PBM payload")
            packed = np.frombuffer(payload[:needed], dtype=np.uint8).reshape(
                height, row_bytes
            )
            return np.unpackbits(packed, axis=1, bitorder="big")[:, :width] == 0
        if maxval < 256:
            if len(payload) < count:
                raise CodecError("truncated PNM payload")
            array = np.frombuffer(payload[:count], dtype=np.uint8).copy()
        else:
            if len(payload) < count * 2:
                raise CodecError("truncated 16-bit PNM payload")
            source = np.frombuffer(payload[: count * 2], dtype=np.uint8)
            native = np.empty(count * 2, dtype=np.uint8)
            if lib().mio_swap16(
                addr(source), source.nbytes, addr(native), native.nbytes, count
            ):
                raise CodecError("native PNM byte swap rejected buffer metadata")
            array = native.view(np.uint16)
    if magic == b"P1":
        array = array == 0
    elif maxval not in (255, 65535):
        target = 255 if maxval < 256 else 65535
        array = (
            (array.astype(np.uint64) * target + maxval // 2) // maxval
        ).astype(np.uint8 if target == 255 else np.uint16)
    shape = (height, width, 3) if channels == 3 else (height, width)
    return array.reshape(shape)


def encode_pnm(image, kind: str, **kwargs) -> bytes:
    if kwargs:
        raise TypeError(f"unsupported PNM option: {next(iter(kwargs))}")
    array = np.asarray(image)
    normalized = kind.lower()
    if normalized == "pnm":
        normalized = "ppm" if array.ndim == 3 else "pgm"
    if normalized == "pbm":
        if array.ndim != 2:
            raise ValueError("PBM expects HxW image data")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("PBM samples must be binary")
        height, width = array.shape
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        bits = np.packbits(array == 0, axis=1, bitorder="big")
        return f"P4\n{width} {height}\n".encode() + bits.tobytes()
    channels = 3 if normalized == "ppm" else 1
    if channels == 1 and array.ndim != 2:
        raise ValueError("PGM expects HxW image data")
    if channels == 3 and (array.ndim != 3 or array.shape[2] != 3):
        raise ValueError("PPM expects HxWx3 image data")
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise TypeError("PGM/PPM supports uint8 and uint16 images")
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    maxval = 255 if array.dtype == np.uint8 else 65535
    magic = "P6" if channels == 3 else "P5"
    header = f"{magic}\n{width} {height}\n{maxval}\n".encode()
    contiguous = np.ascontiguousarray(array)
    if array.dtype == np.uint8:
        return header + contiguous.tobytes()
    source = contiguous.view(np.uint8).reshape(-1)
    swapped = np.empty_like(source)
    if lib().mio_swap16(
        addr(source), source.nbytes, addr(swapped), swapped.nbytes, array.size
    ):
        raise CodecError("native PNM byte swap rejected buffer metadata")
    return header + swapped.tobytes()


def encode_qoi(image, **kwargs) -> bytes:
    colorspace = kwargs.pop("colorspace", 0)
    if kwargs:
        raise TypeError(f"unsupported QOI option: {next(iter(kwargs))}")
    if colorspace not in (0, 1):
        raise ValueError("QOI colorspace must be 0 or 1")
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
        raise TypeError("QOI expects HxWx3 or HxWx4 uint8 image data")
    height, width, channels = array.shape
    if not 0 < width <= 0xFFFFFFFF or not 0 < height <= 0xFFFFFFFF:
        raise ValueError("QOI dimensions must fit uint32 and be positive")
    source = np.ascontiguousarray(array).reshape(-1)
    encoded = np.empty(width * height * 5, dtype=np.uint8)
    index = np.empty(256, dtype=np.uint8)
    size = lib().mio_qoi_encode(
        addr(source),
        source.nbytes,
        addr(encoded),
        encoded.nbytes,
        addr(index),
        index.nbytes,
        width * height,
        channels,
    )
    if size < 0 or size > encoded.nbytes:
        raise CodecError("native QOI encoder rejected buffer metadata")
    header = struct.pack(">4sIIBB", b"qoif", width, height, channels, colorspace)
    return header + encoded[:size].tobytes() + QOI_PADDING


def decode_qoi(data: bytes, **kwargs) -> np.ndarray:
    if kwargs:
        raise TypeError(f"unsupported QOI option: {next(iter(kwargs))}")
    if len(data) < 22:
        raise CodecError("truncated QOI image")
    magic, width, height, channels, _colorspace = struct.unpack_from(
        ">4sIIBB", data
    )
    if magic != b"qoif" or width == 0 or height == 0 or channels not in (3, 4):
        raise CodecError("invalid QOI header")
    if data[-8:] != QOI_PADDING:
        raise CodecError("invalid QOI end marker")
    body = data[14:-8]
    if width * height > len(body) * 62:
        raise CodecError("QOI opcode stream cannot contain the declared pixels")
    source = _u8_buffer(body)
    decoded = np.empty(width * height * channels, dtype=np.uint8)
    index = np.empty(256, dtype=np.uint8)
    consumed = lib().mio_qoi_decode(
        addr(source),
        source.nbytes,
        addr(decoded),
        decoded.nbytes,
        addr(index),
        index.nbytes,
        len(body),
        width * height,
        channels,
    )
    if consumed < 0 or consumed != len(body):
        raise CodecError("invalid QOI opcode stream")
    return decoded.reshape(height, width, channels)
