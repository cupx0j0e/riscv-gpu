#!/usr/bin/env python3
"""
Live stroke/digit classifier using the 4x4 systolic array over UART.

UI (pygame):
- Draw with left mouse; release to run inference on the hardware (runs in a worker thread to avoid UI hangs).
- 'c' clears the canvas, 'q' quits.
- Shows last prediction on screen (digit if using --mnist-json, otherwise gesture).

Requires: pyserial, pygame. Install with: pip install pyserial pygame
"""

import argparse
import sys
import time
from typing import List, Tuple, Optional
import threading

import pygame

from stroke_demo import (
    SystolicArrayClient,
    rasterize_points,
    build_demo_weights,
    run_model,
    load_tinytiny_json,
    run_tinytiny_linear,
    features_dxdy,
    quantize_dxdy,
    render_stroke_to_grid,
)


def normalize_points(points: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[float, float]]:
    return [(x / max(1, w - 1), y / max(1, h - 1)) for (x, y) in points]


def draw_ui(screen, font, prediction: str, logits):
    txt = f"Pred: {prediction}   Logits: {logits}"
    surf = font.render(txt, True, (255, 255, 255))
    screen.blit(surf, (10, 10))
    help_txt = "Draw with mouse. Release to infer. 'c' clear, 'q' quit."
    screen.blit(font.render(help_txt, True, (180, 180, 180)), (10, 30))


def main(argv):
    parser = argparse.ArgumentParser(description="Live stroke classifier on 4x4 systolic array (UART)")
    parser.add_argument("--port", required=True, help="UART port (e.g. /dev/ttyUSB1)")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud (default 115200)")
    parser.add_argument("--len16", action="store_true", help="force 2-byte length framing")
    parser.add_argument("--no-setlen", action="store_true", help="skip SET_LEN (use firmware defaults)")
    parser.add_argument("--stream-len", type=int, default=8, help="stream_len register (default 8)")
    parser.add_argument("--flush-len", type=int, default=10, help="flush_len register (default 10)")
    parser.add_argument("--size", type=int, default=256, help="canvas size (square, default 256)")
    parser.add_argument("--tinytiny-json", help="path to tinytinyTPU gesture_model.json (2x2 linear)")
    parser.add_argument("--mnist-json", help="path to mnist_int8.json (8x8 -> 32 -> 10 MLP)")
    args = parser.parse_args(argv)

    pygame.init()
    size = (args.size, args.size)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("4x4 Systolic Array Stroke Classifier")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    use_tiny = args.tinytiny_json is not None
    use_mnist = args.mnist_json is not None
    if use_tiny and use_mnist:
        print("Choose only one of --tinytiny-json or --mnist-json")
        return 1

    if use_tiny:
        tiny_model = load_tinytiny_json(args.tinytiny_json)
    elif use_mnist:
        from stroke_demo import load_mnist_json, run_mnist_int8  # avoid circular import at top
        mnist_model = load_mnist_json(args.mnist_json)
    else:
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
                sys.stderr.write(f"SET_LEN rejected ({e}); continuing with firmware defaults\n")
        prediction = "none"
        logits: List[int] = []
        stroke: List[Tuple[int, int]] = []
        drawing = False
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
                        screen.fill((0, 0, 0))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    drawing = True
                    stroke = [event.pos]
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drawing = False
                    if stroke:
                        norm = normalize_points(stroke, *size)
                        if inference_thread and inference_thread.is_alive():
                            prediction = "busy..."
                        else:
                            prediction = "running..."
                            logits = []
                            def worker(points_norm):
                                nonlocal pending_result
                                try:
                                    if use_tiny:
                                        dx, dy = features_dxdy(points_norm)
                                        dxdy_q = quantize_dxdy(dx, dy, tiny_model.in_scale)
                                        res_logits, pred_idx = run_tinytiny_linear(client, dxdy_q, tiny_model)
                                        pred_label = "horizontal" if pred_idx == 0 else "vertical"
                                    elif use_mnist:
                                        gridm = render_stroke_to_grid(points_norm, grid=mnist_model.grid_size, thickness=2, blur=True)
                                        res_logits, pred_idx = run_mnist_int8(client, gridm, mnist_model)
                                        pred_label = str(pred_idx)
                                    else:
                                        grid = rasterize_points(points_norm)
                                        res_logits, pred_idx = run_model(client, grid, weights)
                                        pred_label = "horizontal" if pred_idx == 0 else "vertical"
                                    pending_result = {"pred": pred_label, "logits": res_logits}
                                except Exception as exc:
                                    pending_result = {"pred": f"error: {exc}", "logits": []}
                            inference_thread = threading.Thread(target=worker, args=(norm,), daemon=True)
                            inference_thread.start()
                    else:
                        prediction = "none"
                elif event.type == pygame.MOUSEMOTION and drawing:
                    stroke.append(event.pos)

            # collect finished inference result
            if inference_thread and not inference_thread.is_alive():
                inference_thread = None
                if pending_result is not None:
                    prediction = pending_result.get("pred", "none")
                    logits = pending_result.get("logits", [])
                    pending_result = None

            # draw stroke
            screen.fill((0, 0, 0))
            if len(stroke) > 1:
                pygame.draw.lines(screen, (0, 200, 255), False, stroke, 4)

            draw_ui(screen, font, prediction, logits)
            pygame.display.flip()
            clock.tick(60)
    finally:
        client.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
