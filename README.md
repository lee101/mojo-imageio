# mojo-imageio

`mojo-imageio` is a standalone image encode/decode library whose byte-heavy codec
work runs in compiled [Mojo](https://www.modular.com/mojo). Its Python entry points
mirror the covered part of `imageio`: use `mojo_imageio.v3.imread` and
`mojo_imageio.v3.imwrite`, or the `v2`-shaped top-level functions.

This is a focused port, not a wrapper around upstream imageio. Reading dispatches
from file magic and writing dispatches from the extension. Paths, encoded byte
strings, `BytesIO` objects, and imageio's `"<bytes>"` output convention are
supported.

## Coverage

The native subset is:

- PNG: 8-bit grayscale, grayscale-alpha, RGB, and RGBA encode/decode; 16-bit
  grayscale encode and non-palette 16-bit decode; 8-bit palette decode; all five
  scanline filters and adaptive filtering.
- BMP: uncompressed 8-bit palette, 24-bit RGB, and 32-bit input; grayscale and RGB
  output, including top-down BMP rows.
- Netpbm: ASCII and binary PBM/PGM/PPM (`P1` through `P6`) decode; binary
  PBM/PGM/PPM encode; 8-bit and 16-bit PGM/PPM samples.
- QOI: complete RGB/RGBA lossless encode and decode, including INDEX, DIFF, LUMA,
  RUN, RGB, and RGBA opcodes.
- `v3.imread`, `v3.imwrite`, `v3.imiter`, plus `v2`-style `imread`, `imwrite`,
  `imsave`, `mimread`, and single-image `mimwrite`.

JPEG, GIF, TIFF, WebP, animated or volumetric images, metadata APIs, interlaced
PNG, sub-8-bit PNG samples, compressed BMP variants, and the full set of
format-specific imageio keyword arguments are not covered. Unsupported data is
rejected instead of silently delegated to another library.

## Install and use

The pinned Mojo nightly and all Python dependencies are managed by pixi:

```bash
pixi install
pixi run build
```

The following example runs from the repository after installation:

```python
import numpy as np
import mojo_imageio.v3 as iio

y, x = np.indices((480, 640))
rgb = np.stack(((x + y) & 255, x & 255, y & 255), axis=2).astype(np.uint8)

encoded = iio.imwrite(iio.RETURN_BYTES, rgb, extension=".png")
decoded = iio.imread(encoded)
assert np.array_equal(decoded, rgb)

iio.imwrite("image.qoi", rgb)
```

For the upstream-v2 call shape:

```python
import mojo_imageio as imageio

image = imageio.imread("input.bmp")
imageio.imwrite("output.ppm", image)
```

Run verification and benchmarks with:

```bash
pixi run test
pixi run bench
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(72 logical CPUs). Times are the best of repeated end-to-end calls and include
Python dispatch, allocation, container handling, and compression where applicable.

| operation | mojo-imageio | reference | speedup |
| --- | ---: | ---: | ---: |
| PNG encode 2048x1536 RGB | 138.78 ms | 271.39 ms (imageio) | 1.96x |
| PNG decode 2048x1536 RGB | 52.76 ms | 66.96 ms (imageio) | 1.27x |
| BMP encode 2048x1536 RGB | 7.11 ms | 14.98 ms (imageio) | 2.11x |
| BMP decode 2048x1536 RGB | 6.36 ms | 14.25 ms (imageio) | 2.24x |
| PPM encode 2048x1536 RGB | 2.56 ms | 13.33 ms (imageio) | 5.21x |
| QOI encode 512x512 RGB | 3.93 ms | 629.24 ms (Python reference) | 160.12x |

PNG adaptive filtering scores all five candidates in one SIMD pass and distributes
large, independent scanlines across physical cores. Filter 0 and filter 2 decoding
also use SIMD fast paths. PNG deflate and inflate use the zlib-compatible zlib-ng
backend; NumPy pixel and scanline buffers still cross the FFI boundary without a
copy. BMP and PPM avoid Pillow's object/plugin overhead, and the native QOI state
machine is much faster than the independent Python reference.

The QOI row uses the independent implementation of the published QOI algorithm
in `bench/bench.py`; it is not an upstream-imageio comparison.

## How it works

`src/imageio.mojo` is one compilation unit built as
`dist/libmojo-imageio.so`. Python owns validation, files, PNG chunks, CRCs, and
zlib streams. Contiguous NumPy arrays cross the C ABI as integer addresses through
`ctypes`; Mojo reconstructs mutable `UInt8` pointers and writes only into
caller-allocated buffers.

PNG pixels use interleaved row-major HWC storage. Mojo selects and emits PNG
scanline filters before zlib compression and reverses them after decompression.
BMP kernels reverse bottom-up row order, handle four-byte row padding, and swap
RGB/BGR channels. The PNM path uses Mojo for 16-bit network-order conversion. QOI
keeps its 64-entry pixel index in caller-provided scratch memory, so no allocation
or ownership crosses the FFI boundary.

Tests compare encoded and decoded pixels in both directions against real upstream
imageio for PNG, BMP, PGM, and PPM. QOI additionally uses fixed wire bytes and an
independent decoder.

MIT licensed.
