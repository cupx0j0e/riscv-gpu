#!/usr/bin/env python3
"""
Stroke orientation demo for the 4×4 systolic array UART firmware.

What it does:
- Builds a tiny int8 MLP (16→8→2) that votes for horizontal vs vertical strokes.
- Rasterizes a stroke into a 4×4 grid (int8 values).
- Drives the existing UART RPC (test.c) to run both GEMMs on the array.
- Prints the logits and the predicted class.

Requires: pyserial (`pip install pyserial`), Python 3.8+.
"""

import argparse
import struct
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional
import json
import math
import os

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    serial = None
    _serial_import_error = exc


# UART RPC opcodes/status (must match firmware test.c)
OPC_PING = 0x00
OPC_SET_LEN = 0x01
OPC_LOAD_A = 0x10
OPC_LOAD_B = 0x11
OPC_RUN = 0x20
OPC_READ_C = 0x21
OPC_VERSION = 0x30

STATUS_OK = 0x00
STATUS_BADLEN = 0xEE
STATUS_BADOP = 0xEF


def clamp_int8(x: float) -> int:
    return max(-128, min(127, int(round(x))))


def pack_u32_le(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)


class SystolicArrayClient:
    """Minimal UART client for the systolic firmware RPC."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0,
                 two_byte_len: bool = False) -> None:
        if serial is None:
            raise RuntimeError(f"pyserial not available: {_serial_import_error}")
        self.two_byte_len = two_byte_len
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        self.reset_buffers()

    def close(self) -> None:
        self.ser.close()

    def reset_buffers(self) -> None:
        """Flush any stale bytes to avoid protocol misalignment."""
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def _read_exact(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = self.ser.read(n - len(data))
            if not chunk:
                raise RuntimeError(f"UART timeout after {len(data)}/{n} bytes")
            data.extend(chunk)
        return bytes(data)

    def _send(self, op: int, payload: bytes) -> None:
        frame = self._build_frame(op, payload, self.two_byte_len)
        self.ser.write(frame)
        self.ser.flush()

    @staticmethod
    def _build_frame(op: int, payload: bytes, len16: bool) -> bytes:
        if len16:
            if len(payload) > 0xFFFF:
                raise ValueError("payload too large for 2-byte length")
            return bytes([op]) + struct.pack("<H", len(payload)) + payload
        if len(payload) > 255:
            raise ValueError("payload too large for 1-byte length")
        return bytes([op, len(payload)]) + payload

    def ping(self, retries: int = 3) -> None:
        status = STATUS_BADOP
        for attempt in range(retries):
            self.reset_buffers()
            status = self._send_with_fallback(OPC_PING, b"")
            if status == STATUS_OK:
                return
        raise RuntimeError(f"PING failed after {retries} attempts, status=0x{status:02X}")

    def set_lengths(self, stream_len: int = 8, flush_len: int = 10) -> None:
        payload = pack_u32_le((flush_len << 16) | (stream_len & 0xFFFF))
        status = self._send_with_fallback(OPC_SET_LEN, payload)
        if status != STATUS_OK:
            raise RuntimeError(f"SET_LEN failed, status=0x{status:02X}")

    def _max_tiles_per_frame(self) -> int:
        # Protocol uses a 1-byte length; 255 bytes max payload => 3 tiles (3*64=192).
        return 255 // 64

    def _load_tiles(self, op: int, mats16: Sequence[int]) -> int:
        if len(mats16) == 0 or len(mats16) % 16 != 0:
            raise ValueError("expected 16 * tile_count elements")
        tile_count = len(mats16) // 16
        if tile_count > self._max_tiles_per_frame():
            raise ValueError(f"tile_count {tile_count} exceeds protocol limit {self._max_tiles_per_frame()}")
        payload = b"".join(pack_u32_le(v & 0xFF) for v in mats16)
        status = self._send_with_fallback(op, payload)
        if status != STATUS_OK:
            raise RuntimeError(f"LOAD failed (op=0x{op:02X}), status=0x{status:02X}")
        return tile_count

    def load_a(self, mat16: Sequence[int]) -> None:
        self._load_tiles(OPC_LOAD_A, mat16)

    def load_b(self, mat16: Sequence[int]) -> None:
        self._load_tiles(OPC_LOAD_B, mat16)

    def run(self) -> None:
        status = self._send_with_fallback(OPC_RUN, b"")
        if status != STATUS_OK:
            raise RuntimeError(f"RUN failed, status=0x{status:02X}")

    def read_c_tiles(self, tile_count: int) -> List[int]:
        if tile_count <= 0:
            return []
        self._send(OPC_READ_C, b"")
        resp = self._read_exact(1 + tile_count * 16 * 4)
        status = resp[0]
        if status != STATUS_OK:
            raise RuntimeError(f"READ_C failed, status=0x{status:02X}")
        c_vals = list(struct.unpack("<" + "i" * (tile_count * 16), resp[1:]))
        return c_vals

    def read_c(self) -> List[int]:
        self._send(OPC_READ_C, b"")
        resp = self._read_exact(1 + 16 * 4)
        status = resp[0]
        if status != STATUS_OK:
            raise RuntimeError(f"READ_C failed, status=0x{status:02X}")
        c_vals = list(struct.unpack("<16i", resp[1:]))
        return c_vals

    def _send_with_fallback(self, op: int, payload: bytes) -> int:
        """
        Try the configured length encoding; on BADLEN (0xEE/0xEF) flush,
        flip encoding, and retry once to handle firmware variants.
        """
        for attempt in range(2):
            try:
                self.reset_buffers()
                self._send(op, payload)
                status = self._read_exact(1)[0]
                if status in (STATUS_BADLEN, STATUS_BADOP) and attempt == 0:
                    self.two_byte_len = not self.two_byte_len
                    continue
                return status
            except RuntimeError:
                if attempt == 0:
                    self.two_byte_len = not self.two_byte_len
                    continue
                raise
        return STATUS_BADOP


def rasterize_points(points: Sequence[Tuple[float, float]], grid: int = 4) -> List[int]:
    """Map normalized points (0..1) into a grid of ints (0..127)."""
    if not points:
        return [0] * (grid * grid)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = max(1e-6, xmax - xmin)
    span_y = max(1e-6, ymax - ymin)
    out = [0] * (grid * grid)
    for x, y in points:
        gx = int((x - xmin) / span_x * (grid - 1) + 0.5)
        gy = int((y - ymin) / span_y * (grid - 1) + 0.5)
        gx = max(0, min(grid - 1, gx))
        gy = max(0, min(grid - 1, gy))
        out[gy * grid + gx] = min(127, out[gy * grid + gx] + 8)
    return [clamp_int8(v) for v in out]


def render_stroke_to_grid(points: Sequence[Tuple[float, float]], grid: int,
                          thickness: int = 1, supersample: int = 4, border_frac: float = 0.1,
                          intensity_step: int = 32, center_canvas: bool = True) -> List[int]:
    """
    Rasterize a stroke into a high-res canvas (grid*supersample),
    center/scale it, then downsample to grid x grid (0..127 values).
    """
    if not points:
        return [0] * (grid * grid)

    ss = max(2, supersample)
    hi = grid * ss
    canvas = [[0 for _ in range(hi)] for _ in range(hi)]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = max(1e-6, xmax - xmin)
    span_y = max(1e-6, ymax - ymin)
    span = max(span_x, span_y)
    fill_frac = max(0.6, min(0.9, 1.0 - 2 * border_frac))
    target_span = fill_frac * hi
    scale = target_span / span
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    def to_grid(pt: Tuple[float, float]) -> Tuple[float, float]:
        gx = (pt[0] - cx) * scale + (hi - 1) / 2.0
        gy = (pt[1] - cy) * scale + (hi - 1) / 2.0
        return gx, gy

    def stamp_circle(cx_px: float, cy_px: float, radius: float) -> None:
        r = max(0.0, radius)
        r_int = max(1, int(math.ceil(r)))
        r2 = r * r
        y0 = max(0, int(math.floor(cy_px - r_int)))
        y1 = min(hi - 1, int(math.ceil(cy_px + r_int)))
        x0 = max(0, int(math.floor(cx_px - r_int)))
        x1 = min(hi - 1, int(math.ceil(cx_px + r_int)))
        for yy in range(y0, y1 + 1):
            dy = yy - cy_px
            for xx in range(x0, x1 + 1):
                dx = xx - cx_px
                if dx * dx + dy * dy <= r2:
                    val = canvas[yy][xx] + intensity_step
                    canvas[yy][xx] = 255 if val > 255 else val

    radius = max(0.5, float(thickness))
    for i in range(len(points)):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        x0, y0 = to_grid(p0)
        x1, y1 = to_grid(p1)
        dx = x1 - x0
        dy = y1 - y0
        seg_len = max(1.0, math.hypot(dx, dy))
        steps = int(seg_len) + 1
        for s in range(steps + 1):
            t = s / steps if steps else 0.0
            cx_px = x0 + dx * t
            cy_px = y0 + dy * t
            stamp_circle(cx_px, cy_px, radius)

    # Downsample by block average to mimic MNIST-like blur
    block = hi // grid
    flat: List[int] = []
    max_val = 0
    for gy in range(grid):
        for gx in range(grid):
            acc = 0
            count = 0
            for yy in range(gy * block, min((gy + 1) * block, hi)):
                for xx in range(gx * block, min((gx + 1) * block, hi)):
                    acc += canvas[yy][xx]
                    count += 1
            val = acc // max(1, count)
            flat.append(val)
            if val > max_val:
                max_val = val

    if max_val > 0:
        flat = [min(127, max(0, int(round(v * 127.0 / max_val)))) for v in flat]
    else:
        flat = [0] * (grid * grid)
    return flat


def generate_demo_stroke(kind: str, n: int = 32) -> List[Tuple[float, float]]:
    """Synthetic strokes for quick testing."""
    if kind not in ("horizontal", "vertical"):
        raise ValueError("kind must be 'horizontal' or 'vertical'")
    pts = []
    if kind == "horizontal":
        y = 0.5
        for i in range(n):
            x = i / max(1, n - 1)
            pts.append((x, y + 0.02 * (-1) ** i))
    else:
        x = 0.5
        for i in range(n):
            y = i / max(1, n - 1)
            pts.append((x + 0.02 * (-1) ** i, y))
    return pts


# ---------------------------------------------------------------------------
# tinytinyTPU-style 2x2 linear model loader (uint8 zp=128 export)
# ---------------------------------------------------------------------------

@dataclass
class TinyLinearModel:
    w_int8: List[List[int]]  # 2 x 2
    b_int: List[int]         # scaled int32 bias
    in_scale: float
    w_scale: float
    zero_point: int


@dataclass
class MnistLayer:
    w: List[List[int]]  # int8, shape out x in (exported), we will transpose
    b: List[int]        # int32
    scale_in: float
    scale_w: float
    scale_out: float
    zp_in: int = 0
    zp_w: int = 0
    zp_out: int = 0


@dataclass
class MnistModel:
    layers: List[MnistLayer]
    input_scale: float
    input_zp: int
    input_dim: int
    grid_size: int


def load_tinytiny_json(path: str) -> TinyLinearModel:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w_q = data["weights"]
    b_f = data["bias"]
    zp = int(data.get("zero_point", 128))
    w_scale = float(data.get("scale", 127.0))
    if len(w_q) != 2 or len(w_q[0]) != 2:
        raise ValueError("expected weights shape 2x2 in tinytiny JSON")
    # Convert uint8 with zp->int8
    w_int8 = [[int(w_q[r][c]) - zp for c in range(2)] for r in range(2)]
    # Input scale: map dx/dy in [0,1] to int8
    in_scale = 127.0
    # Bias in int32 domain: bias_float * (in_scale * w_scale)
    bias_int = [int(round(b * in_scale * w_scale)) for b in b_f]
    return TinyLinearModel(w_int8=w_int8, b_int=bias_int, in_scale=in_scale, w_scale=w_scale, zero_point=zp)


def features_dxdy(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Return normalized dx, dy in [0,1] from a stroke."""
    if not points:
        return 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return (dx, dy)


def quantize_dxdy(dx: float, dy: float, scale: float = 127.0) -> List[int]:
    return [clamp_int8(dx * scale), clamp_int8(dy * scale)]


# MNIST int8 export loader (from train_mini_mnist.py)
def load_mnist_json(path: str) -> MnistModel:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layers = []
    for l in data["layers"]:
        layers.append(MnistLayer(
            w=l["w"],
            b=l["b"],
            scale_in=l["scale_in"],
            scale_w=l["scale_w"],
            scale_out=l["scale_out"],
            zp_in=l.get("zp_in", 0),
            zp_w=l.get("zp_w", 0),
            zp_out=l.get("zp_out", 0),
        ))
    input_dim = len(layers[0].w[0]) if layers and layers[0].w else 64
    grid = int(math.isqrt(input_dim))
    if grid * grid != input_dim:
        raise ValueError(f"Input dim {input_dim} not a perfect square; cannot infer grid size")
    return MnistModel(
        layers=layers,
        input_scale=data.get("input_scale", 1.0/127.0),
        input_zp=data.get("input_zp", 0),
        input_dim=input_dim,
        grid_size=grid,
    )


@dataclass
class ModelWeights:
    w1: List[List[int]]  # 16 x 8
    b1: List[int]        # 8
    w2: List[List[int]]  # 8 x 2
    b2: List[int]        # 2
    act1_scale: float = 8.0  # scale to quantize layer1 outputs to int8


def build_demo_weights() -> ModelWeights:
    """Hand-coded weights: first layer builds row/col sums, second picks class."""
    w1 = [[0 for _ in range(8)] for _ in range(16)]
    for r in range(4):
        for c in range(4):
            w1[r * 4 + c][r] = 1
    for c in range(4):
        for r in range(4):
            w1[r * 4 + c][4 + c] = 1
    b1 = [0] * 8
    w2 = [[0, 0] for _ in range(8)]
    for f in range(4):
        w2[f][0] = 1
    for f in range(4, 8):
        w2[f][1] = 1
    b2 = [0, 0]
    return ModelWeights(w1=w1, b1=b1, w2=w2, b2=b2, act1_scale=8.0)


def run_gemm_1xk_kxn(client: SystolicArrayClient, a_vec: Sequence[int],
                     b_mat: Sequence[Sequence[int]]) -> List[int]:
    """Compute 1×K by K×N using the 4×4 array with tiling (multi-tile batches)."""
    K = len(a_vec)
    if K == 0:
        return []
    N = len(b_mat[0])
    for row in b_mat:
        if len(row) != N:
            raise ValueError("inconsistent B matrix width")
    out = [0] * N
    max_tiles = client._max_tiles_per_frame()
    for n0 in range(0, N, 4):
        partial = [0, 0, 0, 0]
        tiles_a: List[int] = []
        tiles_b: List[int] = []
        pending_tiles = 0

        def flush_batch() -> None:
            nonlocal tiles_a, tiles_b, pending_tiles, partial
            if pending_tiles == 0:
                return
            client._load_tiles(OPC_LOAD_A, tiles_a)
            client._load_tiles(OPC_LOAD_B, tiles_b)
            client.run()
            c_vals = client.read_c_tiles(pending_tiles)
            for t in range(pending_tiles):
                base = t * 16
                for c in range(4):
                    partial[c] += c_vals[base + c]
            tiles_a = []
            tiles_b = []
            pending_tiles = 0

        for k0 in range(0, K, 4):
            tile_a = [0] * 16
            for j in range(4):
                idx = k0 + j
                tile_a[j] = clamp_int8(a_vec[idx]) if idx < K else 0
            tile_b = [0] * 16
            for r in range(4):
                for c in range(4):
                    kk = k0 + r
                    nn = n0 + c
                    if kk < K and nn < N:
                        tile_b[r * 4 + c] = clamp_int8(b_mat[kk][nn])
            tiles_a.extend(tile_a)
            tiles_b.extend(tile_b)
            pending_tiles += 1
            if pending_tiles >= max_tiles:
                flush_batch()
        flush_batch()
        for c in range(4):
            if n0 + c < N:
                out[n0 + c] = partial[c]
    return out


def relu_int32(vals: Iterable[int]) -> List[int]:
    return [v if v > 0 else 0 for v in vals]


def quantize_int8(vals: Iterable[int], scale: float) -> List[int]:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return [clamp_int8(v / scale) for v in vals]


def run_model(client: SystolicArrayClient, grid: Sequence[int],
              weights: ModelWeights) -> Tuple[List[int], int]:
    """Runs the 2-layer int8 MLP and returns (logits, predicted_class)."""
    l1_acc = run_gemm_1xk_kxn(client, grid, weights.w1)
    l1_acc = [l1_acc[i] + weights.b1[i] for i in range(len(l1_acc))]
    l1_act = relu_int32(l1_acc)
    l1_q = quantize_int8(l1_act, weights.act1_scale)
    l2_acc = run_gemm_1xk_kxn(client, l1_q, weights.w2)
    l2_acc = [l2_acc[i] + weights.b2[i] for i in range(len(l2_acc))]
    pred = 0 if l2_acc[0] >= l2_acc[1] else 1
    return l2_acc, pred


def run_tinytiny_linear(client: SystolicArrayClient, dxdy_q: Sequence[int],
                        model: TinyLinearModel) -> Tuple[List[int], int]:
    """Run 1x2 * 2x2 -> 1x2 for tinytinyTPU export (uint8 zp=128 converted to int8)."""
    logits = run_gemm_1xk_kxn(client, dxdy_q, model.w_int8)
    logits = [logits[i] + model.b_int[i] for i in range(2)]
    pred = 0 if logits[0] >= logits[1] else 1
    return logits, pred


def run_mnist_int8(client: SystolicArrayClient, img: Sequence[int], model: MnistModel) -> Tuple[List[int], int]:
    """Run flattened grayscale through int8 model (size inferred from weights)."""
    if len(img) != model.input_dim:
        raise ValueError(f"expected {model.input_dim} values, got {len(img)}")

    # Quantize input
    act = [clamp_int8(round((v / 127.0) / model.input_scale)) for v in img]
    for idx, layer in enumerate(model.layers):
        w = layer.w  # out x in
        if not w:
            raise ValueError("empty weight matrix")
        if len(act) != len(w[0]):
            raise ValueError(f"layer {idx} expected input len {len(w[0])}, got {len(act)}")
        # Fallback: run per 4x4 tile using existing helper (no batching)
        # Transpose weights to KxN
        out_dim = len(w)
        in_dim = len(w[0])
        w_t = [[w_row[c] for w_row in w] for c in range(in_dim)]  # K x N
        logits = run_gemm_1xk_kxn(client, act, w_t)
        logits = [logits[i] + layer.b[i] for i in range(len(logits))]
        if idx == len(model.layers) - 1:
            act = logits
        else:
            relu = [v if v > 0 else 0 for v in logits]
            s_out = layer.scale_out if layer.scale_out > 0 else 1e-6
            act = [clamp_int8(round(v / s_out)) for v in relu]
    pred = max(range(len(act)), key=lambda i: act[i])
    return act, pred


def parse_points_file(path: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"bad line in {path!r}: {line!r}")
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Stroke orientation demo on 4x4 systolic array")
    parser.add_argument("--port", required=True, help="UART port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud (default: 115200)")
    parser.add_argument("--len16", action="store_true",
                        help="use 2-byte length field (if your firmware expects it)")
    parser.add_argument("--no-setlen", action="store_true",
                        help="skip SET_LEN if your firmware rejects it (defaults are 8/10)")
    parser.add_argument("--stream-len", type=int, default=8, help="stream_len register (default 8)")
    parser.add_argument("--flush-len", type=int, default=10, help="flush_len register (default 10)")
    parser.add_argument("--tinytiny-json", help="path to tinytinyTPU gesture_model.json (2x2 linear)")
    parser.add_argument("--mnist-json", help="path to mnist_int8.json (8x8 -> 32 -> 10 MLP)")
    parser.add_argument("--mnist-idx", type=int, help="if set, run this MNIST test image instead of drawn stroke")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", choices=["horizontal", "vertical"], default="horizontal",
                       help="use a synthetic stroke (default: horizontal)")
    group.add_argument("--points", help="path to text file of x y pairs (normalized or raw)")
    args = parser.parse_args(argv)

    if serial is None:
        sys.stderr.write("pyserial not installed; pip install pyserial\n")
        return 1

    if args.points:
        pts_raw = parse_points_file(args.points)
    else:
        pts_raw = generate_demo_stroke(args.demo)

    # Normalize points to [0,1] for feature extraction
    xs = [p[0] for p in pts_raw] or [0.0]
    ys = [p[1] for p in pts_raw] or [0.0]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = max(1e-6, xmax - xmin)
    span_y = max(1e-6, ymax - ymin)
    pts_norm = [((p[0] - xmin) / span_x, (p[1] - ymin) / span_y) for p in pts_raw]

    use_tiny = args.tinytiny_json is not None
    use_mnist = args.mnist_json is not None
    if use_tiny and use_mnist:
        sys.stderr.write("Choose only one of --tinytiny-json or --mnist-json\n")
        return 1

    if use_tiny:
        model = load_tinytiny_json(args.tinytiny_json)
        dx, dy = features_dxdy(pts_norm)
        dxdy_q = quantize_dxdy(dx, dy, model.in_scale)
        print(f"dx, dy (norm): {dx:.3f}, {dy:.3f}  -> int8 {dxdy_q}")
    elif use_mnist:
        mnist_model = load_mnist_json(args.mnist_json)
        if args.mnist_idx is not None:
            try:
                import torch
                import torchvision.transforms as T
                from torchvision.datasets import MNIST
            except ImportError:
                sys.stderr.write("torch/torchvision not installed; pip install torch torchvision\n")
                return 1
            xfm = T.Compose([T.Resize((mnist_model.grid_size, mnist_model.grid_size)), T.ToTensor()])
            ds = MNIST(root="./data", train=False, download=True, transform=xfm)
            img, label = ds[args.mnist_idx]
            grid_m = (img[0] * 127).round().clamp(0, 127).to(torch.int).view(-1).tolist()
            print(f"Using MNIST test idx {args.mnist_idx} (true label {label}) -> grid generated.")
        else:
            grid_m = render_stroke_to_grid(pts_norm, grid=mnist_model.grid_size, thickness=2, center_canvas=True)
            print(f"{mnist_model.grid_size}x{mnist_model.grid_size} grid (row-major):", grid_m)
    else:
        grid = rasterize_points(pts_norm)
        print("Grid values (row-major, int8):", grid)
        weights = build_demo_weights()
    client = SystolicArrayClient(args.port, baud=args.baud, two_byte_len=args.len16)
    try:
        client.ping()
        if args.no_setlen:
            print("Skipping SET_LEN (using firmware defaults)")
        else:
            try:
                client.set_lengths(stream_len=args.stream_len, flush_len=args.flush_len)
            except RuntimeError as e:
                msg = str(e)
                if "0xEE" in msg or "0xEF" in msg:
                    sys.stderr.write(f"SET_LEN rejected ({msg}); continuing with firmware defaults\n")
                else:
                    raise
        if use_tiny:
            logits, pred = run_tinytiny_linear(client, dxdy_q, model)
            label = "horizontal" if pred == 0 else "vertical"
            print(f"Logits: {logits}, predicted: {label}")
        elif use_mnist:
            logits, pred = run_mnist_int8(client, grid_m, mnist_model)
            print(f"Logits: {logits}, predicted digit: {pred}")
        else:
            logits, pred = run_model(client, grid, weights)
            label = "horizontal" if pred == 0 else "vertical"
            print(f"Logits: {logits}, predicted: {label}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
