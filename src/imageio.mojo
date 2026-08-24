"""Native byte transforms for image codecs exposed through a C ABI."""

from std.math import abs
from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime PNG_PARALLEL_BYTES = 262144


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = abs(p - a)
    var pb = abs(p - b)
    var pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def residual_score(value: Int) -> Int:
    var wrapped = value & 255
    if wrapped >= 128:
        return 256 - wrapped
    return wrapped


@always_inline
def residual_vector[W: Int](
    raw: SIMD[DType.uint8, W], prediction: SIMD[DType.uint8, W]
) -> SIMD[DType.int64, W]:
    var difference = raw - prediction
    var zero = SIMD[DType.uint8, W](0)
    return min(difference, zero - difference).cast[DType.int64]()


@always_inline
def average_vector[W: Int](
    left: SIMD[DType.uint8, W], up: SIMD[DType.uint8, W]
) -> SIMD[DType.uint8, W]:
    return (left & up) + ((left ^ up) >> 1)


@always_inline
def paeth_vector[W: Int](
    left_u8: SIMD[DType.uint8, W],
    up_u8: SIMD[DType.uint8, W],
    upper_left_u8: SIMD[DType.uint8, W],
) -> SIMD[DType.uint8, W]:
    var left = left_u8.cast[DType.int16]()
    var up = up_u8.cast[DType.int16]()
    var upper_left = upper_left_u8.cast[DType.int16]()
    var estimate = left + up - upper_left
    var left_distance = abs(estimate - left)
    var up_distance = abs(estimate - up)
    var upper_left_distance = abs(estimate - upper_left)
    var use_left = left_distance.le(up_distance) & left_distance.le(
        upper_left_distance
    )
    return use_left.select(
        left, up_distance.le(upper_left_distance).select(up, upper_left)
    ).cast[DType.uint8]()


def png_filter_row(
    src: BPtr, dst: BPtr, y: Int, stride: Int, bpp: Int, mode: Int
):
    comptime W = simdwidthof[DType.float64]()
    var row = y * stride
    var dst_row = y * (stride + 1)
    var best_filter = mode
    if mode < 0:
        var score0 = SIMD[DType.int64, W](0)
        var score1 = SIMD[DType.int64, W](0)
        var score2 = SIMD[DType.int64, W](0)
        var score3 = SIMD[DType.int64, W](0)
        var score4 = SIMD[DType.int64, W](0)
        var scalar0 = 0
        var scalar1 = 0
        var scalar2 = 0
        var scalar3 = 0
        var scalar4 = 0
        var x = 0
        var prefix = min(bpp, stride)
        while x < prefix:
            var raw = Int(src[row + x])
            var up = Int(src[row + x - stride]) if y > 0 else 0
            scalar0 += residual_score(raw)
            scalar1 += residual_score(raw)
            scalar2 += residual_score(raw - up)
            scalar3 += residual_score(raw - up // 2)
            scalar4 += residual_score(raw - up)
            x += 1
        while x + W <= stride:
            var raw = src.load[width=W](row + x)
            var left = src.load[width=W](row + x - bpp)
            var up = (
                src.load[width=W](row + x - stride)
                if y > 0
                else SIMD[DType.uint8, W](0)
            )
            var upper_left = (
                src.load[width=W](row + x - stride - bpp)
                if y > 0
                else SIMD[DType.uint8, W](0)
            )
            score0 += residual_vector(raw, SIMD[DType.uint8, W](0))
            score1 += residual_vector(raw, left)
            score2 += residual_vector(raw, up)
            score3 += residual_vector(raw, average_vector(left, up))
            score4 += residual_vector(
                raw, paeth_vector(left, up, upper_left)
            )
            x += W
        while x < stride:
            var raw = Int(src[row + x])
            var left = Int(src[row + x - bpp])
            var up = Int(src[row + x - stride]) if y > 0 else 0
            var upper_left = (
                Int(src[row + x - stride - bpp]) if y > 0 else 0
            )
            scalar0 += residual_score(raw)
            scalar1 += residual_score(raw - left)
            scalar2 += residual_score(raw - up)
            scalar3 += residual_score(raw - (left + up) // 2)
            scalar4 += residual_score(raw - paeth(left, up, upper_left))
            x += 1
        scalar0 += Int(score0.reduce_add())
        scalar1 += Int(score1.reduce_add())
        scalar2 += Int(score2.reduce_add())
        scalar3 += Int(score3.reduce_add())
        scalar4 += Int(score4.reduce_add())
        var best_score = scalar0
        best_filter = 0
        if scalar1 < best_score:
            best_score = scalar1
            best_filter = 1
        if scalar2 < best_score:
            best_score = scalar2
            best_filter = 2
        if scalar3 < best_score:
            best_score = scalar3
            best_filter = 3
        if scalar4 < best_score:
            best_filter = 4

    dst[dst_row] = UInt8(best_filter)
    var x = 0
    var prefix = min(bpp, stride)
    while x < prefix:
        var raw = Int(src[row + x])
        var up = Int(src[row + x - stride]) if y > 0 else 0
        var prediction = 0
        if best_filter == 2:
            prediction = up
        elif best_filter == 3:
            prediction = up // 2
        elif best_filter == 4:
            prediction = up
        dst[dst_row + 1 + x] = UInt8((raw - prediction) & 255)
        x += 1
    while x + W <= stride:
        var raw = src.load[width=W](row + x)
        var left = src.load[width=W](row + x - bpp)
        var up = (
            src.load[width=W](row + x - stride)
            if y > 0
            else SIMD[DType.uint8, W](0)
        )
        var upper_left = (
            src.load[width=W](row + x - stride - bpp)
            if y > 0
            else SIMD[DType.uint8, W](0)
        )
        var prediction = SIMD[DType.uint8, W](0)
        if best_filter == 1:
            prediction = left
        elif best_filter == 2:
            prediction = up
        elif best_filter == 3:
            prediction = average_vector(left, up)
        elif best_filter == 4:
            prediction = paeth_vector(left, up, upper_left)
        dst.store(dst_row + 1 + x, raw - prediction)
        x += W
    while x < stride:
        var raw = Int(src[row + x])
        var left = Int(src[row + x - bpp])
        var up = Int(src[row + x - stride]) if y > 0 else 0
        var upper_left = (
            Int(src[row + x - stride - bpp]) if y > 0 else 0
        )
        var prediction = 0
        if best_filter == 1:
            prediction = left
        elif best_filter == 2:
            prediction = up
        elif best_filter == 3:
            prediction = (left + up) // 2
        elif best_filter == 4:
            prediction = paeth(left, up, upper_left)
        dst[dst_row + 1 + x] = UInt8((raw - prediction) & 255)
        x += 1


async def png_filter_worker(
    src: BPtr,
    dst: BPtr,
    height: Int,
    stride: Int,
    bpp: Int,
    mode: Int,
    worker: Int,
    workers: Int,
):
    var y0 = worker * height // workers
    var y1 = (worker + 1) * height // workers
    for y in range(y0, y1):
        png_filter_row(src, dst, y, stride, bpp, mode)


def png_filter(src: BPtr, dst: BPtr, height: Int, stride: Int, bpp: Int, mode: Int):
    var workers = (
        min(height, num_physical_cores())
        if height * stride >= PNG_PARALLEL_BYTES
        else 1
    )

    if workers > 1:
        initialize_runtime()
        var tasks = TaskGroup()
        for worker in range(workers):
            tasks.create_task(
                png_filter_worker(
                    src, dst, height, stride, bpp, mode, worker, workers
                )
            )
        tasks.wait()
    else:
        png_filter_row(src, dst, 0, stride, bpp, mode)
        for y in range(1, height):
            png_filter_row(src, dst, y, stride, bpp, mode)


def png_unfilter(src: BPtr, dst: BPtr, height: Int, stride: Int, bpp: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    for y in range(height):
        var src_row = y * (stride + 1)
        var row = y * stride
        var filter_type = Int(src[src_row])
        if filter_type < 0 or filter_type > 4:
            return -1
        if filter_type == 0:
            var x = 0
            while x + W <= stride:
                dst.store(row + x, src.load[width=W](src_row + 1 + x))
                x += W
            while x < stride:
                dst[row + x] = src[src_row + 1 + x]
                x += 1
            continue
        if filter_type == 2:
            var x = 0
            while x + W <= stride:
                var up = (
                    dst.load[width=W](row + x - stride)
                    if y > 0
                    else SIMD[DType.uint8, W](0)
                )
                dst.store(
                    row + x, src.load[width=W](src_row + 1 + x) + up
                )
                x += W
            while x < stride:
                var up = Int(dst[row + x - stride]) if y > 0 else 0
                dst[row + x] = UInt8(
                    (Int(src[src_row + 1 + x]) + up) & 255
                )
                x += 1
            continue
        for x in range(stride):
            var filtered = Int(src[src_row + 1 + x])
            var left = Int(dst[row + x - bpp]) if x >= bpp else 0
            var up = Int(dst[row + x - stride]) if y > 0 else 0
            var upper_left = (
                Int(dst[row + x - stride - bpp])
                if y > 0 and x >= bpp
                else 0
            )
            var prediction = 0
            if filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = up
            elif filter_type == 3:
                prediction = (left + up) // 2
            elif filter_type == 4:
                prediction = paeth(left, up, upper_left)
            dst[row + x] = UInt8((filtered + prediction) & 255)
    return 0


def bmp_pack(
    src: BPtr, dst: BPtr, width: Int, height: Int, channels: Int, dst_stride: Int
):
    for y in range(height):
        var src_row = y * width * channels
        var dst_row = (height - 1 - y) * dst_stride
        if channels == 1:
            for x in range(width):
                dst[dst_row + x] = src[src_row + x]
        else:
            for x in range(width):
                var s = src_row + x * channels
                var d = dst_row + x * channels
                dst[d] = src[s + 2]
                dst[d + 1] = src[s + 1]
                dst[d + 2] = src[s]
                if channels == 4:
                    dst[d + 3] = src[s + 3]
        for x in range(width * channels, dst_stride):
            dst[dst_row + x] = UInt8(0)


def bmp_unpack(
    src: BPtr,
    dst: BPtr,
    width: Int,
    height: Int,
    channels: Int,
    src_stride: Int,
    top_down: Bool,
):
    for y in range(height):
        var stored_y = y if top_down else height - 1 - y
        var src_row = stored_y * src_stride
        var dst_row = y * width * channels
        if channels == 1:
            for x in range(width):
                dst[dst_row + x] = src[src_row + x]
        else:
            for x in range(width):
                var s = src_row + x * channels
                var d = dst_row + x * channels
                dst[d] = src[s + 2]
                dst[d + 1] = src[s + 1]
                dst[d + 2] = src[s]
                if channels == 4:
                    dst[d + 3] = src[s + 3]


def swap16(src: BPtr, dst: BPtr, count: Int):
    for i in range(count):
        dst[2 * i] = src[2 * i + 1]
        dst[2 * i + 1] = src[2 * i]


def qoi_hash(r: Int, g: Int, b: Int, a: Int) -> Int:
    return (r * 3 + g * 5 + b * 7 + a * 11) & 63


def index_matches(index: BPtr, slot: Int, r: Int, g: Int, b: Int, a: Int) -> Bool:
    var p = slot * 4
    return (
        Int(index[p]) == r
        and Int(index[p + 1]) == g
        and Int(index[p + 2]) == b
        and Int(index[p + 3]) == a
    )


def index_store(index: BPtr, slot: Int, r: Int, g: Int, b: Int, a: Int):
    var p = slot * 4
    index[p] = UInt8(r)
    index[p + 1] = UInt8(g)
    index[p + 2] = UInt8(b)
    index[p + 3] = UInt8(a)


def qoi_encode(
    src: BPtr, dst: BPtr, index: BPtr, pixels: Int, channels: Int
) -> Int:
    for i in range(256):
        index[i] = UInt8(0)
    var pr = 0
    var pg = 0
    var pb = 0
    var pa = 255
    var run = 0
    var pos = 0
    for i in range(pixels):
        var s = i * channels
        var r = Int(src[s])
        var g = Int(src[s + 1])
        var b = Int(src[s + 2])
        var a = Int(src[s + 3]) if channels == 4 else 255
        if r == pr and g == pg and b == pb and a == pa:
            run += 1
            if run == 62 or i == pixels - 1:
                dst[pos] = UInt8(0xC0 | (run - 1))
                pos += 1
                run = 0
        else:
            if run > 0:
                dst[pos] = UInt8(0xC0 | (run - 1))
                pos += 1
                run = 0
            var slot = qoi_hash(r, g, b, a)
            if index_matches(index, slot, r, g, b, a):
                dst[pos] = UInt8(slot)
                pos += 1
            else:
                index_store(index, slot, r, g, b, a)
                if a == pa:
                    var dr = r - pr
                    var dg = g - pg
                    var db = b - pb
                    var dr_dg = dr - dg
                    var db_dg = db - dg
                    if (
                        dr >= -2 and dr <= 1
                        and dg >= -2 and dg <= 1
                        and db >= -2 and db <= 1
                    ):
                        dst[pos] = UInt8(
                            0x40 | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2)
                        )
                        pos += 1
                    elif (
                        dg >= -32 and dg <= 31
                        and dr_dg >= -8 and dr_dg <= 7
                        and db_dg >= -8 and db_dg <= 7
                    ):
                        dst[pos] = UInt8(0x80 | (dg + 32))
                        dst[pos + 1] = UInt8((dr_dg + 8) << 4 | (db_dg + 8))
                        pos += 2
                    else:
                        dst[pos] = UInt8(0xFE)
                        dst[pos + 1] = UInt8(r)
                        dst[pos + 2] = UInt8(g)
                        dst[pos + 3] = UInt8(b)
                        pos += 4
                else:
                    dst[pos] = UInt8(0xFF)
                    dst[pos + 1] = UInt8(r)
                    dst[pos + 2] = UInt8(g)
                    dst[pos + 3] = UInt8(b)
                    dst[pos + 4] = UInt8(a)
                    pos += 5
        pr = r
        pg = g
        pb = b
        pa = a
    return pos


def qoi_decode(
    src: BPtr, dst: BPtr, index: BPtr, src_len: Int, pixels: Int, channels: Int
) -> Int:
    for i in range(256):
        index[i] = UInt8(0)
    var r = 0
    var g = 0
    var b = 0
    var a = 255
    var run = 0
    var pos = 0
    for i in range(pixels):
        if run > 0:
            run -= 1
        else:
            if pos >= src_len:
                return -1
            var tag = Int(src[pos])
            pos += 1
            if tag == 0xFE:
                if pos + 3 > src_len:
                    return -1
                r = Int(src[pos])
                g = Int(src[pos + 1])
                b = Int(src[pos + 2])
                pos += 3
            elif tag == 0xFF:
                if pos + 4 > src_len:
                    return -1
                r = Int(src[pos])
                g = Int(src[pos + 1])
                b = Int(src[pos + 2])
                a = Int(src[pos + 3])
                pos += 4
            elif (tag & 0xC0) == 0x00:
                var p = (tag & 63) * 4
                r = Int(index[p])
                g = Int(index[p + 1])
                b = Int(index[p + 2])
                a = Int(index[p + 3])
            elif (tag & 0xC0) == 0x40:
                r = (r + ((tag >> 4) & 3) - 2) & 255
                g = (g + ((tag >> 2) & 3) - 2) & 255
                b = (b + (tag & 3) - 2) & 255
            elif (tag & 0xC0) == 0x80:
                if pos >= src_len:
                    return -1
                var extra = Int(src[pos])
                pos += 1
                var dg = (tag & 63) - 32
                r = (r + dg + ((extra >> 4) & 15) - 8) & 255
                g = (g + dg) & 255
                b = (b + dg + (extra & 15) - 8) & 255
            else:
                run = tag & 63
        index_store(index, qoi_hash(r, g, b, a), r, g, b, a)
        var d = i * channels
        dst[d] = UInt8(r)
        dst[d + 1] = UInt8(g)
        dst[d + 2] = UInt8(b)
        if channels == 4:
            dst[d + 3] = UInt8(a)
    return pos


@export("mio_png_filter")
def mio_png_filter(
    src: Int, src_len: Int, dst: Int, dst_len: Int,
    height: Int, stride: Int, bpp: Int, mode: Int
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or height <= 0 or stride <= 0
        or bpp <= 0 or bpp > stride or mode < -1 or mode > 4
        or src_len < 0 or dst_len < 0
        or stride > src_len or stride >= dst_len
        or height > src_len // stride
        or height > dst_len // (stride + 1)
    ):
        return -1
    png_filter(bp(src), bp(dst), height, stride, bpp, mode)
    return 0


@export("mio_png_unfilter")
def mio_png_unfilter(
    src: Int, src_len: Int, dst: Int, dst_len: Int,
    height: Int, stride: Int, bpp: Int
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or height <= 0 or stride <= 0
        or bpp <= 0 or bpp > stride or src_len < 0 or dst_len < 0
        or stride >= src_len or stride > dst_len
        or height > src_len // (stride + 1)
        or height > dst_len // stride
    ):
        return -1
    return png_unfilter(bp(src), bp(dst), height, stride, bpp)


@export("mio_bmp_pack")
def mio_bmp_pack(
    src: Int, src_len: Int, dst: Int, dst_len: Int,
    width: Int, height: Int, channels: Int, dst_stride: Int
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or width <= 0 or height <= 0
        or (channels != 1 and channels != 3 and channels != 4)
        or src_len < 0 or dst_len < 0
        or width > src_len // channels or width > dst_len // channels
        or dst_stride < width * channels
        or height > src_len // (width * channels)
        or height > dst_len // dst_stride
    ):
        return -1
    bmp_pack(bp(src), bp(dst), width, height, channels, dst_stride)
    return 0


@export("mio_bmp_unpack")
def mio_bmp_unpack(
    src: Int,
    src_len: Int,
    dst: Int,
    dst_len: Int,
    width: Int,
    height: Int,
    channels: Int,
    src_stride: Int,
    top_down: Int,
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or width <= 0 or height <= 0
        or (channels != 1 and channels != 3 and channels != 4)
        or src_len < 0 or dst_len < 0
        or width > src_len // channels or width > dst_len // channels
        or src_stride < width * channels
        or height > src_len // src_stride
        or height > dst_len // (width * channels)
    ):
        return -1
    bmp_unpack(
        bp(src), bp(dst), width, height, channels, src_stride, top_down != 0
    )
    return 0


@export("mio_swap16")
def mio_swap16(
    src: Int, src_len: Int, dst: Int, dst_len: Int, count: Int
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or count <= 0 or src_len < 0 or dst_len < 0
        or count > src_len // 2 or count > dst_len // 2
    ):
        return -1
    swap16(bp(src), bp(dst), count)
    return 0


@export("mio_qoi_encode")
def mio_qoi_encode(
    src: Int, src_len: Int, dst: Int, dst_len: Int,
    index: Int, index_len: Int, pixels: Int, channels: Int
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or index == 0 or pixels <= 0
        or (channels != 3 and channels != 4)
        or src_len < 0 or dst_len < 0 or index_len < 256
        or pixels > src_len // channels or pixels > dst_len // 5
    ):
        return -1
    return qoi_encode(bp(src), bp(dst), bp(index), pixels, channels)


@export("mio_qoi_decode")
def mio_qoi_decode(
    src: Int,
    src_buffer_len: Int,
    dst: Int,
    dst_len: Int,
    index: Int,
    index_len: Int,
    src_len: Int,
    pixels: Int,
    channels: Int,
) abi("C") -> Int:
    if (
        src == 0 or dst == 0 or index == 0 or src_len <= 0 or pixels <= 0
        or (channels != 3 and channels != 4)
        or src_buffer_len < src_len or dst_len < 0 or index_len < 256
        or pixels > dst_len // channels
    ):
        return -1
    return qoi_decode(bp(src), bp(dst), bp(index), src_len, pixels, channels)
