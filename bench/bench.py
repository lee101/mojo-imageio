"""End-to-end codec benchmarks against imageio and an independent QOI encoder."""

from __future__ import annotations

import os
import platform
import sys
import time

import imageio.v3 as upstream
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojo_imageio.v3 as iio  # noqa: E402


def time_call(fn, repetitions=5):
    fn()
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def reference_qoi_encode(image):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)
    index = [(0, 0, 0, 0)] * 64
    previous = (0, 0, 0, 255)
    run = 0
    body = bytearray()
    for number, sample in enumerate(pixels):
        pixel = tuple(int(value) for value in sample)
        if channels == 3:
            pixel += (255,)
        if pixel == previous:
            run += 1
            if run == 62 or number == len(pixels) - 1:
                body.append(0xC0 | (run - 1))
                run = 0
        else:
            if run:
                body.append(0xC0 | (run - 1))
                run = 0
            r, g, b, a = pixel
            slot = (r * 3 + g * 5 + b * 7 + a * 11) % 64
            if index[slot] == pixel:
                body.append(slot)
            else:
                index[slot] = pixel
                pr, pg, pb, pa = previous
                if a == pa:
                    dr, dg, db = r - pr, g - pg, b - pb
                    dr_dg, db_dg = dr - dg, db - dg
                    if -2 <= dr <= 1 and -2 <= dg <= 1 and -2 <= db <= 1:
                        body.append(
                            0x40 | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2)
                        )
                    elif (
                        -32 <= dg <= 31
                        and -8 <= dr_dg <= 7
                        and -8 <= db_dg <= 7
                    ):
                        body.extend(
                            (0x80 | (dg + 32), (dr_dg + 8) << 4 | (db_dg + 8))
                        )
                    else:
                        body.extend((0xFE, r, g, b))
                else:
                    body.extend((0xFF, r, g, b, a))
        previous = pixel
    header = (
        b"qoif"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((channels, 0))
    )
    return header + body + b"\x00\x00\x00\x00\x00\x00\x00\x01"


def machine_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def display_row(name, mojo_seconds, reference_seconds, reference_name):
    speedup = reference_seconds / mojo_seconds
    comparison = f"{speedup:.2f}x"
    if speedup < 1:
        comparison += " (slower)"
    print(
        f"| {name} | {mojo_seconds * 1000:.2f} ms | "
        f"{reference_seconds * 1000:.2f} ms ({reference_name}) | {comparison} |"
    )


def main():
    y, x = np.indices((1536, 2048), dtype=np.uint32)
    image = np.empty((1536, 2048, 3), dtype=np.uint8)
    image[:, :, 0] = (x // 3 + y // 7) & 255
    image[:, :, 1] = (x // 11 + y // 2) & 255
    image[:, :, 2] = ((x ^ y) // 5) & 255

    iio.imwrite("<bytes>", image[:2], extension=".png")
    upstream.imwrite("<bytes>", image[:2], extension=".png")

    rows = []
    mojo_time, mojo_png = time_call(
        lambda: iio.imwrite("<bytes>", image, extension=".png"), repetitions=3
    )
    upstream_time, upstream_png = time_call(
        lambda: upstream.imwrite("<bytes>", image, extension=".png"), repetitions=3
    )
    assert np.array_equal(iio.imread(upstream_png), image)
    assert np.array_equal(upstream.imread(mojo_png, extension=".png"), image)
    rows.append(("PNG encode 2048x1536 RGB", mojo_time, upstream_time, "imageio"))

    mojo_time, actual = time_call(lambda: iio.imread(upstream_png))
    upstream_time, reference = time_call(lambda: upstream.imread(upstream_png))
    assert np.array_equal(actual, reference)
    rows.append(("PNG decode 2048x1536 RGB", mojo_time, upstream_time, "imageio"))

    mojo_time, mojo_bmp = time_call(
        lambda: iio.imwrite("<bytes>", image, extension=".bmp")
    )
    upstream_time, upstream_bmp = time_call(
        lambda: upstream.imwrite("<bytes>", image, extension=".bmp")
    )
    assert np.array_equal(iio.imread(upstream_bmp), image)
    assert np.array_equal(upstream.imread(mojo_bmp), image)
    rows.append(("BMP encode 2048x1536 RGB", mojo_time, upstream_time, "imageio"))

    mojo_time, actual = time_call(lambda: iio.imread(upstream_bmp))
    upstream_time, reference = time_call(lambda: upstream.imread(upstream_bmp))
    assert np.array_equal(actual, reference)
    rows.append(("BMP decode 2048x1536 RGB", mojo_time, upstream_time, "imageio"))

    mojo_time, mojo_ppm = time_call(
        lambda: iio.imwrite("<bytes>", image, extension=".ppm")
    )
    upstream_time, upstream_ppm = time_call(
        lambda: upstream.imwrite("<bytes>", image, extension=".ppm")
    )
    assert np.array_equal(iio.imread(upstream_ppm), image)
    assert np.array_equal(upstream.imread(mojo_ppm), image)
    rows.append(("PPM encode 2048x1536 RGB", mojo_time, upstream_time, "imageio"))

    qoi_image = image[:512, :512].copy()
    mojo_time, mojo_qoi = time_call(
        lambda: iio.imwrite("<bytes>", qoi_image, extension=".qoi")
    )
    reference_time, reference_qoi = time_call(
        lambda: reference_qoi_encode(qoi_image), repetitions=3
    )
    assert mojo_qoi == reference_qoi
    rows.append(
        ("QOI encode 512x512 RGB", mojo_time, reference_time, "Python reference")
    )

    print(f"Machine: {machine_name()}, {os.cpu_count()} logical CPUs")
    print()
    print("| operation | mojo-imageio | reference | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        display_row(*row)


if __name__ == "__main__":
    main()
