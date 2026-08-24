import io
import struct

import imageio.v3 as upstream
import numpy as np
import pytest
from PIL import Image

import mojo_imageio as imageio
import mojo_imageio.v3 as iio
from mojo_imageio._codecs import CodecError
from mojo_imageio._lib import lib

rng = np.random.default_rng(2026)


@pytest.mark.parametrize(
    "shape",
    [(19, 31), (19, 31, 2), (19, 31, 3), (19, 31, 4)],
)
def test_png_roundtrip_all_uint8_color_types(shape):
    expected = rng.integers(0, 256, size=shape, dtype=np.uint8)
    encoded = iio.imwrite("<bytes>", expected, extension=".png")
    assert np.array_equal(iio.imread(encoded), expected)


@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
def test_png_every_filter_decodes_like_upstream(filter_type):
    expected = rng.integers(0, 256, size=(23, 37, 4), dtype=np.uint8)
    encoded = iio.imwrite(
        "<bytes>", expected, extension=".png", filter_type=filter_type
    )
    assert np.array_equal(iio.imread(encoded), expected)
    assert np.array_equal(upstream.imread(encoded, extension=".png"), expected)


@pytest.mark.parametrize("filter_type", [0, 2, 4])
def test_png_simd_filters_handle_scalar_tail(filter_type):
    expected = rng.integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    encoded = iio.imwrite(
        "<bytes>", expected, extension=".png", filter_type=filter_type
    )
    assert np.array_equal(iio.imread(encoded), expected)
    assert np.array_equal(upstream.imread(encoded, extension=".png"), expected)


@pytest.mark.parametrize("height", [339, 343])
def test_png_adaptive_filter_parallel_threshold(height):
    expected = rng.integers(0, 256, size=(height, 257, 3), dtype=np.uint8)
    encoded = iio.imwrite(
        "<bytes>", expected, extension=".png", compress_level=0
    )
    assert np.array_equal(iio.imread(encoded), expected)
    assert np.array_equal(upstream.imread(encoded, extension=".png"), expected)


@pytest.mark.parametrize(
    ("shape", "extension"),
    [
        ((17, 29), ".png"),
        ((17, 29, 3), ".png"),
        ((17, 29, 4), ".png"),
        ((17, 29), ".bmp"),
        ((17, 29, 3), ".bmp"),
        ((17, 29), ".pgm"),
        ((17, 29, 3), ".ppm"),
    ],
)
def test_decodes_upstream_output(shape, extension):
    expected = rng.integers(0, 256, size=shape, dtype=np.uint8)
    encoded = upstream.imwrite("<bytes>", expected, extension=extension)
    assert np.array_equal(iio.imread(encoded), expected)


@pytest.mark.parametrize(
    ("shape", "extension"),
    [
        ((17, 29), ".png"),
        ((17, 29, 3), ".png"),
        ((17, 29, 4), ".png"),
        ((17, 29), ".bmp"),
        ((17, 29, 3), ".bmp"),
        ((17, 29), ".pgm"),
        ((17, 29, 3), ".ppm"),
    ],
)
def test_upstream_decodes_mojo_output(shape, extension):
    expected = rng.integers(0, 256, size=shape, dtype=np.uint8)
    encoded = iio.imwrite("<bytes>", expected, extension=extension)
    actual = upstream.imread(encoded, extension=extension)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("extension", [".png", ".pgm"])
def test_uint16_grayscale_cross_implementation(extension):
    expected = rng.integers(0, 65536, size=(21, 33), dtype=np.uint16)
    ours = iio.imwrite("<bytes>", expected, extension=extension)
    theirs = upstream.imwrite("<bytes>", expected, extension=extension)
    assert np.array_equal(iio.imread(theirs), expected)
    assert np.array_equal(upstream.imread(ours, extension=extension), expected)


@pytest.mark.parametrize("channels", [2, 3, 4])
def test_uint16_multichannel_png_decode(channels):
    expected = rng.integers(0, 65536, size=(11, 17, channels), dtype=np.uint16)
    raw = expected.astype(">u2").view(np.uint8).reshape(11, -1)
    filtered = b"".join(b"\x00" + row.tobytes() for row in raw)

    def chunk(kind, payload):
        import binascii
        import zlib

        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(
            ">I", binascii.crc32(body) & 0xFFFFFFFF
        )

    import zlib

    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB", 17, 11, 16, {2: 4, 3: 2, 4: 6}[channels], 0, 0, 0
            ),
        )
        + chunk(b"IDAT", zlib.compress(filtered))
        + chunk(b"IEND", b"")
    )
    assert np.array_equal(iio.imread(encoded), expected)


@pytest.mark.filterwarnings("ignore:Palette images with Transparency")
def test_palette_png_matches_upstream():
    indices = np.arange(35, dtype=np.uint8).reshape(5, 7) % 4
    image = Image.fromarray(indices, mode="P")
    palette = [255, 0, 0, 0, 255, 0, 0, 0, 255, 20, 30, 40] + [0] * (768 - 12)
    image.putpalette(palette)
    image.info["transparency"] = bytes((255, 128, 64, 0))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = stream.getvalue()
    assert np.array_equal(iio.imread(encoded), upstream.imread(encoded))
    assert np.array_equal(
        iio.imread(encoded, mode="RGBA"), upstream.imread(encoded, mode="RGBA")
    )


def test_png_crc_corruption_is_rejected():
    expected = np.arange(100, dtype=np.uint8).reshape(10, 10)
    encoded = bytearray(iio.imwrite("<bytes>", expected, extension=".png"))
    encoded[29] ^= 1
    with pytest.raises(CodecError, match="CRC"):
        iio.imread(encoded)


def _reference_qoi_decode(data):
    magic, width, height, channels, _ = struct.unpack_from(">4sIIBB", data)
    assert magic == b"qoif"
    index = [(0, 0, 0, 0)] * 64
    pixel = (0, 0, 0, 255)
    result = []
    pos = 14
    run = 0
    while len(result) < width * height:
        if run:
            run -= 1
        else:
            tag = data[pos]
            pos += 1
            r, g, b, a = pixel
            if tag == 0xFE:
                r, g, b = data[pos : pos + 3]
                pos += 3
            elif tag == 0xFF:
                r, g, b, a = data[pos : pos + 4]
                pos += 4
            elif tag >> 6 == 0:
                r, g, b, a = index[tag]
            elif tag >> 6 == 1:
                r = (r + ((tag >> 4) & 3) - 2) & 255
                g = (g + ((tag >> 2) & 3) - 2) & 255
                b = (b + (tag & 3) - 2) & 255
            elif tag >> 6 == 2:
                extra = data[pos]
                pos += 1
                dg = (tag & 63) - 32
                r = (r + dg + (extra >> 4) - 8) & 255
                g = (g + dg) & 255
                b = (b + dg + (extra & 15) - 8) & 255
            else:
                run = tag & 63
            pixel = (r, g, b, a)
        slot = (pixel[0] * 3 + pixel[1] * 5 + pixel[2] * 7 + pixel[3] * 11) % 64
        index[slot] = pixel
        result.append(pixel[:channels])
    return np.asarray(result, dtype=np.uint8).reshape(height, width, channels)


def test_qoi_published_wire_format_for_one_red_pixel():
    expected = np.array([[[255, 0, 0]]], dtype=np.uint8)
    encoded = iio.imwrite("<bytes>", expected, extension=".qoi")
    assert encoded == (
        b"qoif\x00\x00\x00\x01\x00\x00\x00\x01\x03\x00"
        b"\xfe\xff\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01"
    )
    assert np.array_equal(iio.imread(encoded), expected)


def test_qoi_decoder_covers_every_opcode_family():
    body = bytes(
        (
            0xC0,
            0x76,
            0xA5,
            0x5F,
            0xFF,
            10,
            20,
            30,
            40,
            0xFE,
            50,
            60,
            70,
            12,
            0xC1,
        )
    )
    encoded = (
        b"qoif\x00\x00\x00\x08\x00\x00\x00\x01\x04\x00"
        + body
        + b"\x00\x00\x00\x00\x00\x00\x00\x01"
    )
    expected = np.array(
        [
            (0, 0, 0, 255),
            (1, 255, 0, 255),
            (3, 4, 12, 255),
            (10, 20, 30, 40),
            (50, 60, 70, 40),
            (10, 20, 30, 40),
            (10, 20, 30, 40),
            (10, 20, 30, 40),
        ],
        dtype=np.uint8,
    ).reshape(1, 8, 4)
    assert np.array_equal(iio.imread(encoded), expected)


@pytest.mark.parametrize("channels", [3, 4])
def test_qoi_encoder_matches_independent_decoder(channels):
    expected = rng.integers(0, 256, size=(61, 73, channels), dtype=np.uint8)
    expected[10:30] = expected[9]
    encoded = iio.imwrite("<bytes>", expected, extension=".qoi")
    assert np.array_equal(_reference_qoi_decode(encoded), expected)
    assert np.array_equal(iio.imread(encoded), expected)


def test_bmp_top_down_rows():
    expected = rng.integers(0, 256, size=(7, 11, 3), dtype=np.uint8)
    encoded = bytearray(iio.imwrite("<bytes>", expected, extension=".bmp"))
    offset = struct.unpack_from("<I", encoded, 10)[0]
    height = struct.unpack_from("<i", encoded, 22)[0]
    stride = (expected.shape[1] * 3 + 3) & ~3
    rows = [
        encoded[offset + y * stride : offset + (y + 1) * stride]
        for y in range(height)
    ]
    encoded[offset:] = b"".join(reversed(rows))
    struct.pack_into("<i", encoded, 22, -height)
    assert np.array_equal(iio.imread(encoded), expected)


def test_bmp_32_bit_input_discards_unused_alpha():
    expected = rng.integers(0, 256, size=(4, 7, 3), dtype=np.uint8)
    height, width, _ = expected.shape
    bgra = np.concatenate(
        (expected[:, :, ::-1], np.full((height, width, 1), 73, dtype=np.uint8)),
        axis=2,
    )[::-1]
    offset = 54
    pixels = bgra.tobytes()
    encoded = (
        struct.pack("<2sIHHI", b"BM", offset + len(pixels), 0, 0, offset)
        + struct.pack(
            "<IiiHHIIiiII", 40, width, height, 1, 32, 0, len(pixels), 0, 0, 0, 0
        )
        + pixels
    )
    assert np.array_equal(iio.imread(encoded), expected)


def test_bmp_non_identity_palette_input():
    indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    image = Image.fromarray(indices, mode="P")
    image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 759)
    stream = io.BytesIO()
    image.save(stream, format="BMP")
    assert np.array_equal(iio.imread(stream.getvalue()), upstream.imread(stream.getvalue()))


def test_ascii_pnm_comments_and_non_255_maxval_scales_like_upstream():
    encoded = b"P3\n# generated fixture\n3 1\n15\n0 1 2  3 4 5  # x\n13 14 15\n"
    assert np.array_equal(iio.imread(encoded), upstream.imread(encoded))


@pytest.mark.parametrize(
    "encoded",
    [
        b"P1\n3 1\n0 1 0\n",
        b"P2\n3 1\n15\n0 7 15\n",
        b"P3\n1 1\n15\n0 7 15\n",
        b"P4\n3 1\n\x40",
        b"P5\n3 1\n15\n\x00\x07\x0f",
        b"P6\n1 1\n15\n\x00\x07\x0f",
    ],
)
def test_all_pnm_variants_match_upstream(encoded):
    assert np.array_equal(iio.imread(encoded), upstream.imread(encoded))


def test_uint16_ppm_roundtrip():
    expected = rng.integers(0, 65536, size=(9, 13, 3), dtype=np.uint16)
    encoded = iio.imwrite("<bytes>", expected, extension=".ppm")
    assert np.array_equal(iio.imread(encoded), expected)


def test_pbm_binary_roundtrip():
    expected = np.indices((13, 19)).sum(axis=0) % 3 == 0
    encoded = iio.imwrite("<bytes>", expected, extension=".pbm")
    assert encoded.startswith(b"P4\n")
    assert np.array_equal(iio.imread(encoded), expected)
    assert np.array_equal(upstream.imread(encoded), expected)


def test_file_path_and_file_object_dispatch(tmp_path):
    expected = rng.integers(0, 256, size=(12, 15, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    assert iio.imwrite(path, expected) is None
    assert np.array_equal(iio.imread(path), expected)
    stream = io.BytesIO()
    assert iio.imwrite(stream, expected, extension=".ppm") is None
    stream.seek(0)
    assert np.array_equal(iio.imread(stream), expected)


def test_short_file_object_write_is_not_silently_accepted():
    class ShortWriter:
        def write(self, data):
            return len(data) - 1

    expected = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(OSError, match="short image write"):
        iio.imwrite(ShortWriter(), expected, extension=".png")


def test_v2_names_and_return_bytes():
    expected = rng.integers(0, 256, size=(5, 8), dtype=np.uint8)
    encoded = imageio.imwrite(imageio.RETURN_BYTES, expected, format="PNG")
    assert np.array_equal(imageio.imread(encoded), expected)
    assert np.array_equal(imageio.mimread(encoded)[0], expected)
    assert imageio.mimwrite("<bytes>", [expected], format="PNG") == encoded
    assert imageio.imsave("<bytes>", expected, format="PNG") == encoded


def test_v3_imiter_yields_the_single_image():
    expected = rng.integers(0, 256, size=(5, 8), dtype=np.uint8)
    encoded = iio.imwrite("<bytes>", expected, extension=".png")
    images = list(iio.imiter(encoded))
    assert len(images) == 1
    assert np.array_equal(images[0], expected)


def test_native_exports_reject_invalid_buffer_metadata():
    native = lib()
    assert native.mio_png_filter(0, 0, 0, 0, 1, 1, 1, 0) == -1
    assert native.mio_png_unfilter(0, 0, 0, 0, 1, 1, 1) == -1
    assert native.mio_bmp_pack(0, 0, 0, 0, 1, 1, 3, 4) == -1
    assert native.mio_bmp_unpack(0, 0, 0, 0, 1, 1, 3, 4, 0) == -1
    assert native.mio_swap16(0, 0, 0, 0, 1) == -1
    assert native.mio_qoi_encode(0, 0, 0, 0, 0, 0, 1, 3) == -1
    assert native.mio_qoi_decode(0, 0, 0, 0, 0, 0, 1, 1, 3) == -1


@pytest.mark.parametrize("extension", [".png", ".bmp", ".pgm", ".ppm", ".qoi"])
def test_encoders_reject_silent_dtype_narrowing(extension):
    shape = (2, 3, 3) if extension in (".ppm", ".qoi") else (2, 3)
    with pytest.raises(TypeError):
        iio.imwrite("<bytes>", np.full(shape, 300, dtype=np.int64), extension=extension)


def test_mode_conversion_and_single_image_index():
    gray = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    encoded = iio.imwrite("<bytes>", gray, extension=".png")
    rgb = iio.imread(encoded, mode="RGB", index=0)
    assert rgb.shape == (7, 9, 3)
    assert np.array_equal(rgb[:, :, 0], gray)
    gray_alpha = np.stack((gray, 255 - gray), axis=2)
    ga_encoded = iio.imwrite("<bytes>", gray_alpha, extension=".png")
    rgba = iio.imread(ga_encoded, mode="RGBA")
    assert np.array_equal(rgba[:, :, :3], np.repeat(gray[:, :, None], 3, axis=2))
    assert np.array_equal(rgba[:, :, 3], 255 - gray)
    with pytest.raises(IndexError):
        iio.imread(encoded, index=1)


@pytest.mark.parametrize(
    "data",
    [b"", b"\x89PNG\r\n\x1a\n", b"BM\x00", b"qoif" + b"\x00" * 18, b"P6\n2 2\n255\nx"],
)
def test_truncated_inputs_raise(data):
    with pytest.raises((CodecError, ValueError)):
        iio.imread(data)
