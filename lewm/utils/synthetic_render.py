"""Render synthetic frames from state vectors.

Draws a large white triangle at (x, y) with rotation from angle,
on a black background. No terrain, no decorations — pure signal.

Used to test whether LeWM's JEPA can learn kinematics when the
object is visually salient (large triangle vs tiny lander in real frames).
"""

import math

import numpy as np
from PIL import Image, ImageDraw


def render_synthetic_frame(
    x: float,
    y: float,
    angle: float,
    size: int = 224,
    triangle_radius: int = 35,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Render a single synthetic frame.

    Args:
        x: Normalized x position (roughly [-1, 1]).
        y: Normalized y position (roughly [-0.15, 1.5]).
        angle: Rotation in radians.
        size: Frame size (square).
        triangle_radius: Radius of the triangle in pixels.
        color: RGB color of the triangle.

    Returns:
        (size, size, 3) uint8 array.
    """
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Map state coords to pixel coords
    # x ∈ [-1.5, 1.5] → [0, size], y ∈ [-0.3, 1.5] → [size, 0] (y inverted)
    cx = (x + 1.5) / 3.0 * size
    cy = (1.5 - y) / 1.8 * size

    # Draw triangle rotated by angle (negate for screen coords: +angle = clockwise)
    pts = []
    for k in range(3):
        a = -angle + k * (2 * math.pi / 3) - math.pi / 2
        px = cx + triangle_radius * math.cos(a)
        py = cy + triangle_radius * math.sin(a)
        pts.append((px, py))

    draw.polygon(pts, fill=color, outline=color)

    return np.array(img, dtype=np.uint8)


def render_episode_synthetic(
    states: np.ndarray,
    size: int = 224,
    triangle_radius: int = 35,
) -> np.ndarray:
    """Render an entire episode as synthetic frames.

    Args:
        states: (T, 15) state array. Uses dims 0 (x), 1 (y), 4 (angle).
        size: Frame size.
        triangle_radius: Triangle size in pixels.

    Returns:
        (T, size, size, 3) uint8 array.
    """
    T = len(states)
    frames = np.zeros((T, size, size, 3), dtype=np.uint8)

    for t in range(T):
        x, y = states[t, 0], states[t, 1]
        angle = states[t, 4]
        frames[t] = render_synthetic_frame(x, y, angle, size, triangle_radius)

    return frames
