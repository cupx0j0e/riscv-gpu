#!/usr/bin/env python3
"""
Apply a simple blur to an image by running the 3x3 convolution on the 4x4 systolic
array over UART. Supports small batching of patches when the firmware accepts 2-byte
length fields (use --len16 for that case).

Usage:
  python blur_fpga.py --input path/to/image.png --output blurred.png --port /dev/ttyUSB1 --baud 230400

Requires: pyserial, Pillow. Installs: pip install pyserial pillow
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

from stroke_demo import (  # reuse the existing UART client/protocol
    OPC_LOAD_A,
    OPC_LOAD_B,
    SystolicArrayClient,
    clamp_int8,
)


class BatchingClient(SystolicArrayClient):
    """Allow larger tile batches when using the 2-byte length variant."""

    def _max_tiles_per_frame(self) -> int:
        if getattr(self, "two_byte_len", False):
            # Firmware supports up to 32 tiles; capped by payload length in 2-byte mode.
            return 32
        return super()._max_tiles_per_frame()


def parse_kernel(args_kernel: str, custom_values: Optional[str]) -> List[int]:
    presets = {
        "box3": [1] * 9,
        "gaussian3": [1, 2, 1, 2, 4, 2, 1, 2, 1],
    }
    if custom_values:
        vals = []
        for part in custom_values.replace(",", " ").split():
            vals.append(int(part))
        if len(vals) != 9:
            raise ValueError("custom kernel must have exactly 9 integers for 3x3 blur")
        return vals
    if args_kernel not in presets:
        raise ValueError(f"unknown kernel {args_kernel!r}")
    return presets[args_kernel]


def pack_kernel_tiles(kernel: Sequence[int]) -> List[int]:
    tiles: List[int] = []
    for k0 in range(0, len(kernel), 4):
        tile = [0] * 16
        for r in range(4):
            idx = k0 + r
            if idx < len(kernel):
                tile[r * 4 + 0] = clamp_int8(kernel[idx])
        tiles.extend(tile)
    return tiles


def pack_patch_tiles(patch_q: Sequence[int]) -> List[int]:
    tiles: List[int] = []
    for k0 in range(0, len(patch_q), 4):
        tile = [0] * 16
        for j in range(4):
            idx = k0 + j
            if idx < len(patch_q):
                tile[j] = clamp_int8(patch_q[idx])
        tiles.extend(tile)
    return tiles


def quantize_patch(patch: Sequence[int], scale: float) -> List[int]:
    return [clamp_int8(round(px * scale)) for px in patch]


def dequantize_pixel(acc: int, kernel_sum: int, scale: float) -> int:
    denom = kernel_sum * scale
    if denom <= 0:
        denom = 1.0
    val = int(round(acc / denom))
    return max(0, min(255, val))


def run_patch(client: SystolicArrayClient, patch_q: Sequence[int], kernel_tile_count: int) -> int:
    tiles_a = pack_patch_tiles(patch_q)
    if len(tiles_a) != kernel_tile_count * 16:
        raise ValueError("tile count mismatch between kernel and patch")
    client._load_tiles(OPC_LOAD_A, tiles_a)
    client.run()
    c_vals = client.read_c_tiles(kernel_tile_count)
    acc = 0
    for t in range(kernel_tile_count):
        acc += c_vals[t * 16]  # first row/col accumulates 1xK * Kx1
    return acc


def run_patch_batch(client: SystolicArrayClient, patches_q: Sequence[Sequence[int]],
                    kernel_tile_count: int) -> List[int]:
    if not patches_q:
        return []
    tiles_a: List[int] = []
    for patch in patches_q:
        tiles_a.extend(pack_patch_tiles(patch))
    tile_count = len(tiles_a) // 16
    client._load_tiles(OPC_LOAD_A, tiles_a)
    client.run()
    c_vals = client.read_c_tiles(tile_count)
    outs: List[int] = []
    for p_idx in range(len(patches_q)):
        acc = 0
        base_tile = p_idx * kernel_tile_count
        for t in range(kernel_tile_count):
            acc += c_vals[(base_tile + t) * 16]
        outs.append(acc)
    return outs


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Blur an image using the systolic array over UART")
    parser.add_argument("--input", required=True, help="input image path")
    parser.add_argument("--output", help="output image path (default: <input>_blur.png)")
    parser.add_argument("--port", default="/dev/ttyUSB1", help="UART port (default /dev/ttyUSB1)")
    parser.add_argument("--baud", type=int, default=230400, help="UART baud (default 230400)")
    parser.add_argument("--kernel", choices=["box3", "gaussian3"], default="box3",
                        help="3x3 kernel preset (default box3)")
    parser.add_argument("--kernel-values",
                        help="comma/space separated 9 integers to override the kernel")
    parser.add_argument("--input-scale", type=float, default=127.0 / 255.0,
                        help="scale factor to map 0..255 pixels into int8 (default 127/255)")
    parser.add_argument("--len16", action="store_true",
                        help="use 2-byte length field if your firmware expects it")
    parser.add_argument("--no-setlen", action="store_true",
                        help="skip SET_LEN if your firmware rejects it (defaults are fine)")
    parser.add_argument("--stream-len", type=int, default=8, help="stream_len register (default 8)")
    parser.add_argument("--flush-len", type=int, default=10, help="flush_len register (default 10)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="number of patches per FPGA run (caps at protocol limit; >1 requires firmware that accepts 2-byte length fields with --len16)")
    args = parser.parse_args(argv)

    if Image is None:
        sys.stderr.write("Pillow not installed; pip install pillow\n")
        return 1

    kernel = parse_kernel(args.kernel, args.kernel_values)
    kernel_sum = sum(kernel)
    img_path = Path(args.input)
    out_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + "_blur" + img_path.suffix)

    img = Image.open(img_path).convert("L")
    img = ImageOps.expand(img, border=1, fill=0)  # zero-pad for same-size output
    w_pad, h_pad = img.size
    w, h = w_pad - 2, h_pad - 2
    try:
        pixels = list(img.get_flattened_data())
    except AttributeError:
        pixels = list(img.getdata())

    kernel_tiles = pack_kernel_tiles(kernel)
    kernel_tile_count = len(kernel_tiles) // 16

    client = BatchingClient(args.port, baud=args.baud, two_byte_len=args.len16)
    try:
        if kernel_tile_count > client._max_tiles_per_frame():
            raise ValueError(f"kernel needs {kernel_tile_count} tiles but protocol limit is {client._max_tiles_per_frame()}")
        client.ping()
        if not args.no_setlen:
            try:
                client.set_lengths(stream_len=args.stream_len, flush_len=args.flush_len)
            except RuntimeError as exc:
                msg = str(exc)
                if "0xEE" in msg or "0xEF" in msg:
                    sys.stderr.write(f"SET_LEN rejected ({msg}); continuing with firmware defaults\n")
                else:
                    raise
        client._load_tiles(OPC_LOAD_B, kernel_tiles)

        max_tiles = client._max_tiles_per_frame()
        max_patches_per_batch = max(1, max_tiles // kernel_tile_count)
        batch_goal = max(1, min(args.batch_size, max_patches_per_batch))

        out_vals: List[int] = []
        batch: List[List[int]] = []
        for y in range(h):
            for x in range(w):
                base = (y * w_pad) + x
                patch = [
                    pixels[base + 0], pixels[base + 1], pixels[base + 2],
                    pixels[base + w_pad + 0], pixels[base + w_pad + 1], pixels[base + w_pad + 2],
                    pixels[base + 2 * w_pad + 0], pixels[base + 2 * w_pad + 1],
                    pixels[base + 2 * w_pad + 2],
                ]
                batch.append(quantize_patch(patch, args.input_scale))
                if len(batch) >= batch_goal:
                    accs = run_patch_batch(client, batch, kernel_tile_count)
                    out_vals.extend(dequantize_pixel(acc, kernel_sum, args.input_scale) for acc in accs)
                    batch = []
        if batch:
            accs = run_patch_batch(client, batch, kernel_tile_count)
            out_vals.extend(dequantize_pixel(acc, kernel_sum, args.input_scale) for acc in accs)

        out_img = Image.new("L", (w, h))
        out_img.putdata(out_vals)
        out_img.save(out_path)
        print(f"Saved blurred image to {out_path}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
