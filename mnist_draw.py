#!/usr/bin/env python3
"""
Simplified MNIST-on-systolic demo: draw a digit, we render it MNIST-style (center/blur),
send it to the FPGA, and show the predicted digit. No gesture paths; MNIST only.

Requires: pyserial, pygame. Install with: pip install pyserial pygame
"""

import argparse
import sys
import threading
from typing import List, Tuple, Optional

import pygame

from stroke_demo import (
    SystolicArrayClient,
    load_mnist_json,
    run_mnist_int8,
    render_stroke_to_grid,
)


def normalize_points(points: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[float, float]]:
    return [(x / max(1, w - 1), y / max(1, h - 1)) for (x, y) in points]


def draw_grid_preview(screen, grid: List[int], grid_size: int, scale: int = 6, pos: Tuple[int, int] = (10, 40)) -> None:
    """Visualize the rendered grid in the corner with a border and clamping."""
    x0, y0 = pos
    w = grid_size * scale
    h = grid_size * scale
    pygame.draw.rect(screen, (50, 50, 50), (x0 - 2, y0 - 2, w + 4, h + 4), width=1)
    for r in range(grid_size):
        for c in range(grid_size):
            v = max(0, min(255, grid[r * grid_size + c]))
            col = (v, v, v)
            pygame.draw.rect(screen, col, (x0 + c * scale, y0 + r * scale, scale, scale))


def main(argv):
    parser = argparse.ArgumentParser(description="Draw-and-classify MNIST demo on 4x4 systolic array")
    parser.add_argument("--port", required=True, help="UART port (e.g. /dev/ttyUSB1)")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud (default 115200)")
    parser.add_argument("--len16", action="store_true", help="force 2-byte length framing")
    parser.add_argument("--no-setlen", action="store_true", help="skip SET_LEN (use firmware defaults)")
    parser.add_argument("--stream-len", type=int, default=8, help="stream_len register (default 8)")
    parser.add_argument("--flush-len", type=int, default=10, help="flush_len register (default 10)")
    parser.add_argument("--mnist-json", required=True, help="path to mnist_int8.json")
    parser.add_argument("--mnist-idx", type=int, help="if set, run this MNIST test image instead of drawing")
    parser.add_argument("--size", type=int, default=512, help="canvas size (square, default 512)")
    parser.add_argument("--thickness", type=int, default=0, help="stroke thickness for rendering (0=single pixel core)")
    parser.add_argument("--supersample", type=int, default=4, help="supersample factor before downsampling (default 4)")
    parser.add_argument("--border-frac", type=float, default=0.1, help="border fraction around stroke (default 0.1)")
    parser.add_argument("--intensity-step", type=int, default=32, help="intensity step per plotted point (default 32)")
    parser.add_argument("--dump-grid", help="write rendered grid to this file (CSV) before inference")
    parser.add_argument("--preview-scale", type=int, default=6, help="scale factor for grid preview (default 6)")
    args = parser.parse_args(argv)

    pygame.init()
    size = (args.size, args.size)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("MNIST on Systolic Array")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    mnist_model = load_mnist_json(args.mnist_json)
    client = SystolicArrayClient(args.port, baud=args.baud, two_byte_len=args.len16)
    try:
        client.ping()
        if not args.no_setlen:
            try:
                client.set_lengths(stream_len=args.stream_len, flush_len=args.flush_len)
            except RuntimeError as e:
                sys.stderr.write(f"SET_LEN rejected ({e}); continuing with firmware defaults\n")

        stroke: List[Tuple[int, int]] = []
        drawing = False
        prediction = "none"
        logits: List[int] = []
        grid_last: Optional[List[int]] = None
        inference_thread: Optional[threading.Thread] = None
        pending_result: Optional[dict] = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_c:
                        stroke.clear()
                        prediction = "none"
                        grid_last = None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    drawing = True
                    stroke = [event.pos]
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drawing = False
                    if stroke:
                        if inference_thread and inference_thread.is_alive():
                            prediction = "busy..."
                        else:
                            prediction = "running..."
                            logits = []
                            norm = normalize_points(stroke, *size)
                            gridm = render_stroke_to_grid(
                                norm,
                                grid=mnist_model.grid_size,
                                thickness=args.thickness,
                                supersample=args.supersample,
                                border_frac=args.border_frac,
                                intensity_step=args.intensity_step,
                                center_canvas=True,
                            )
                            grid_last = gridm
                            if args.dump_grid:
                                with open(args.dump_grid, "w", encoding="ascii") as f:
                                    for r in range(mnist_model.grid_size):
                                        row = gridm[r * mnist_model.grid_size:(r + 1) * mnist_model.grid_size]
                                        f.write(",".join(str(v) for v in row) + "\n")
                            with open("grid_dump.txt", "w", encoding="ascii") as f:
                                for r in range(mnist_model.grid_size):
                                    row = gridm[r * mnist_model.grid_size:(r + 1) * mnist_model.grid_size]
                                    f.write(",".join(str(v) for v in row) + "\n")

                            def worker():
                                nonlocal pending_result
                                try:
                                    res_logits, pred_idx = run_mnist_int8(client, gridm, mnist_model)
                                    pending_result = {"pred": str(pred_idx), "logits": res_logits}
                                except Exception as exc:
                                    pending_result = {"pred": f"error: {exc}", "logits": []}

                            inference_thread = threading.Thread(target=worker, daemon=True)
                            inference_thread.start()
                    else:
                        prediction = "none"
                elif event.type == pygame.MOUSEMOTION and drawing:
                    stroke.append(event.pos)

            if inference_thread and not inference_thread.is_alive():
                inference_thread = None
                if pending_result is not None:
                    prediction = pending_result.get("pred", "none")
                    logits = pending_result.get("logits", [])
                    pending_result = None

            # Draw UI
            screen.fill((0, 0, 0))
            if len(stroke) > 1:
                pygame.draw.lines(screen, (0, 200, 255), False, stroke, 4)
            txt = f"Pred: {prediction}  Logits: {logits}"
            screen.blit(font.render(txt, True, (255, 255, 255)), (10, 5))
            screen.blit(font.render("Draw digit, release to infer. 'c' clear, 'q' quit.", True, (180, 180, 180)), (10, 22))
            if grid_last is not None:
                draw_grid_preview(screen, grid_last, mnist_model.grid_size, scale=args.preview_scale, pos=(10, 40))

            pygame.display.flip()
            clock.tick(60)
    finally:
        client.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
