"""Visualization for world model rollout comparisons.

Renders rollout dicts (from rollout_io.py) as:
- Per-dimension time series plots (matplotlib)
- Schematic 2D trajectory videos (PIL + imageio)

Follows the existing video pipeline pattern from visualize_trajectory.py:
PIL for frame rendering, imageio.v3 for MP4 encoding (libx264).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Kinematic dimension names for the first 8 state dims.
KINEMATIC_DIM_NAMES = [
    "x", "y", "vx", "vy", "angle", "angular_vel", "left_leg", "right_leg",
]


def plot_state_overlay(
    rollout: dict,
    output_path: str | Path,
    title: str = "Predicted vs Actual",
    dim_names: list[str] | None = None,
    fps: int = 50,
):
    """Plot per-dimension predicted vs actual state trajectories.

    Args:
        rollout: Rollout dict from rollout_io.run_rollout().
        output_path: Where to save the PNG.
        title: Plot title.
        dim_names: Names for each dimension. Defaults to KINEMATIC_DIM_NAMES.
        fps: Timesteps per second (for x-axis in seconds).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = rollout["predicted_states"]
    actual = rollout["actual_states"]

    if dim_names is None:
        dim_names = KINEMATIC_DIM_NAMES

    T = len(predicted)
    n_dims = min(predicted.shape[1], len(dim_names))
    time_axis = np.arange(T) / fps

    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= n_dims:
            break
        ax.plot(time_axis, actual[:, i], color="#2196F3", linewidth=1.2,
                label="Ground truth", alpha=0.8)
        ax.plot(time_axis, predicted[:, i], color="#F44336", linewidth=1.2,
                label="Predicted", alpha=0.8)
        error = np.abs(predicted[:, i] - actual[:, i])
        ax.fill_between(time_axis, actual[:, i] - error, actual[:, i] + error,
                        color="#F44336", alpha=0.1)
        ax.set_ylabel(dim_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _get_font(size=14):
    """Get a monospace font, falling back gracefully."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            from PIL import ImageFont
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    from PIL import ImageFont
    return ImageFont.load_default()


def _action_bar(value, width=10):
    """Text bar for action value in [-1, 1]."""
    n = int(abs(value) * width)
    if value > 0.05:
        return "=" * n + ">"
    elif value < -0.05:
        return "<" + "=" * n
    else:
        return "-"


def render_trajectory_video(
    rollout: dict,
    output_path: str | Path,
    fps: int = 50,
    canvas_size: tuple[int, int] = (600, 400),
    title: str = "",
    actions: np.ndarray | None = None,
    dim_names: list[str] | None = None,
):
    """Render animated MP4 of predicted vs actual lander trajectory.

    Draws the lander as a triangle at (x, y), rotated by angle, for both
    predicted (red) and ground truth (blue) trajectories. Leaves a fading
    trail of past positions.

    When actions are provided, renders a right-side annotation panel showing
    the action being applied, predicted vs GT state values, and per-step MSE.

    When rollout contains "rgb_frames" (T, H, W, 3) uint8 array, an RGB
    panel is prepended to the left showing the original environment frame.

    Args:
        rollout: Dict with "predicted_states" and "actual_states" arrays.
            Optional "rgb_frames": (T, H, W, 3) uint8 array of original frames.
        output_path: MP4 output path.
        fps: Frames per second (50 matches Box2D physics tick).
        canvas_size: (width, height) of the trajectory view (left panel).
        title: Optional title rendered on each frame.
        actions: (T, action_dim) array of actions applied at each step.
            If provided, a side panel is rendered with action + state info.
        dim_names: Names for state dimensions (for side panel). Defaults
            to KINEMATIC_DIM_NAMES.
    """
    from PIL import Image, ImageDraw
    import imageio.v3 as iio
    import math

    predicted = rollout["predicted_states"]
    actual = rollout["actual_states"]
    rgb_frames = rollout.get("rgb_frames")  # (T, H, W, 3) uint8 or None
    W, H = canvas_size
    T = len(predicted)

    if dim_names is None:
        dim_names = KINEMATIC_DIM_NAMES

    show_panel = actions is not None
    show_rgb = rgb_frames is not None
    # Side panel width — only allocated when we have actions.
    panel_w = 280 if show_panel else 0
    # RGB panel: scale to match canvas height
    rgb_w = 0
    if show_rgb:
        rgb_h_orig, rgb_w_orig = rgb_frames.shape[1], rgb_frames.shape[2]
        rgb_scale = H / rgb_h_orig
        rgb_w = int(rgb_w_orig * rgb_scale)
    total_w = rgb_w + W + panel_w
    # Ensure even dimensions for libx264.
    total_w += total_w % 2
    canvas_h = H + (H % 2)

    # Coordinate mapping: Lunar Lander x ∈ [-1, 1], y ∈ [0, 1.5] roughly.
    # Offset by rgb_w so schematic draws in the right part of the frame.
    def world_to_canvas(x, y):
        cx = rgb_w + int((x + 1.5) / 3.0 * W)
        cy = int((1.5 - y) / 1.8 * H)
        return cx, cy

    def draw_triangle(draw, cx, cy, angle, color, size=12):
        pts = []
        for k in range(3):
            a = angle + k * (2 * math.pi / 3) - math.pi / 2
            px = cx + size * math.cos(a)
            py = cy + size * math.sin(a)
            pts.append((px, py))
        draw.polygon(pts, fill=color, outline=color)

    font = _get_font(14)
    small_font = _get_font(12)
    header_color = (150, 200, 255)
    text_color = (200, 200, 200)
    dim_color = (130, 130, 140)
    gt_color = (33, 150, 243)    # blue
    pred_color = (244, 67, 54)   # red

    frames = []
    trail_actual = []
    trail_pred = []

    for t in range(T):
        img = Image.new("RGB", (total_w, canvas_h), "#1a1a2e")
        draw = ImageDraw.Draw(img)

        # RGB panel (left side, if available).
        if show_rgb and t < len(rgb_frames):
            rgb_img = Image.fromarray(rgb_frames[t]).resize(
                (rgb_w, H), Image.LANCZOS
            )
            img.paste(rgb_img, (0, 0))
            draw.text((4, 4), "RGB", fill="white", font=font)
            # Separator line
            draw.line([(rgb_w - 1, 0), (rgb_w - 1, canvas_h)], fill="#3a3a4a", width=1)

        # Ground line + landing pad.
        gx0, gy = world_to_canvas(-1.5, 0)
        gx1, _ = world_to_canvas(1.5, 0)
        draw.line([(gx0, gy), (gx1, gy)], fill="#4a4a5a", width=2)
        px0, py_pad = world_to_canvas(-0.2, 0)
        px1, _ = world_to_canvas(0.2, 0)
        draw.line([(px0, py_pad), (px1, py_pad)], fill="#FFD700", width=3)

        # Current positions.
        ax, ay = actual[t, 0], actual[t, 1]
        px, py_p = predicted[t, 0], predicted[t, 1]
        trail_actual.append(world_to_canvas(ax, ay))
        trail_pred.append(world_to_canvas(px, py_p))

        # Trails (last 50 positions, fading dots).
        trail_len = min(50, len(trail_actual))
        for k in range(trail_len):
            idx = len(trail_actual) - trail_len + k
            alpha = int(50 + 150 * (k / trail_len))
            tcx, tcy = trail_actual[idx]
            draw.ellipse([(tcx - 2, tcy - 2), (tcx + 2, tcy + 2)],
                         fill=(33, 150, 243, alpha))
            tcx, tcy = trail_pred[idx]
            draw.ellipse([(tcx - 2, tcy - 2), (tcx + 2, tcy + 2)],
                         fill=(244, 67, 54, alpha))

        # Lander triangles.
        acx, acy = world_to_canvas(ax, ay)
        a_angle = actual[t, 4] if actual.shape[1] > 4 else 0
        draw_triangle(draw, acx, acy, a_angle, "#2196F3", size=14)
        pcx, pcy = world_to_canvas(px, py_p)
        p_angle = predicted[t, 4] if predicted.shape[1] > 4 else 0
        draw_triangle(draw, pcx, pcy, p_angle, "#F44336", size=14)

        # Text overlay on trajectory panel.
        n_common = min(predicted.shape[1], actual.shape[1])
        mse = float(np.mean((predicted[t, :n_common] - actual[t, :n_common]) ** 2))
        draw.text((rgb_w + 10, 10), f"t={t:03d}  MSE={mse:.4f}", fill="white", font=font)
        if title:
            draw.text((rgb_w + 10, 28), title, fill="#888888", font=font)
        draw.text((rgb_w + W - 160, H - 40), "GT", fill="#2196F3", font=font)
        draw.text((rgb_w + W - 160, H - 22), "Pred", fill="#F44336", font=font)

        # --- Side panel ---
        if show_panel:
            # Vertical separator line.
            draw.line([(rgb_w + W, 0), (rgb_w + W, canvas_h)], fill="#3a3a4a", width=1)

            x0 = rgb_w + W + 10
            y_pos = 8
            line_h = 18
            small_line_h = 16

            # Step counter.
            draw.text((x0, y_pos), f"Step {t}/{T-1}",
                      fill=(255, 255, 255), font=font)
            y_pos += line_h + 4

            # Action section.
            draw.text((x0, y_pos), "Action", fill=header_color, font=font)
            y_pos += line_h
            if t > 0 and t - 1 < len(actions):
                act = actions[t - 1]
                main_val = float(act[0])
                main_active = main_val > 0
                main_c = (255, 200, 100) if main_active else dim_color
                draw.text((x0, y_pos),
                          f" main: {_action_bar(main_val)}",
                          fill=main_c, font=small_font)
                y_pos += small_line_h
                if len(act) > 1:
                    side_val = float(act[1])
                    side_active = abs(side_val) > 0.5
                    side_c = (100, 200, 255) if side_active else dim_color
                    side_dir = "R" if side_val > 0.5 else "L" if side_val < -0.5 else "-"
                    draw.text((x0, y_pos),
                              f" side: {side_dir} {_action_bar(side_val)}",
                              fill=side_c, font=small_font)
                    y_pos += small_line_h
            else:
                draw.text((x0, y_pos), " (initial)", fill=dim_color, font=small_font)
                y_pos += small_line_h
            y_pos += 4

            # State comparison: GT vs Pred side by side.
            draw.text((x0, y_pos), "State", fill=header_color, font=font)
            y_pos += line_h
            # Column headers.
            draw.text((x0 + 50, y_pos), "GT", fill=gt_color, font=small_font)
            draw.text((x0 + 115, y_pos), "Pred", fill=pred_color, font=small_font)
            draw.text((x0 + 190, y_pos), "Err", fill=dim_color, font=small_font)
            y_pos += small_line_h

            n_show = min(len(dim_names), predicted.shape[1], actual.shape[1])
            for d in range(n_show):
                gt_val = float(actual[t, d])
                pr_val = float(predicted[t, d])
                err = abs(pr_val - gt_val)
                name = dim_names[d][:5]
                # Color error: green if small, yellow if medium, red if large.
                if err < 0.05:
                    err_c = (100, 200, 100)
                elif err < 0.2:
                    err_c = (255, 200, 100)
                else:
                    err_c = (255, 100, 100)
                draw.text((x0, y_pos), f" {name:>5s}", fill=text_color, font=small_font)
                draw.text((x0 + 50, y_pos), f"{gt_val:+.2f}", fill=gt_color, font=small_font)
                draw.text((x0 + 115, y_pos), f"{pr_val:+.2f}", fill=pred_color, font=small_font)
                draw.text((x0 + 190, y_pos), f"{err:.3f}", fill=err_c, font=small_font)
                y_pos += small_line_h
            y_pos += 4

            # Cumulative MSE.
            draw.text((x0, y_pos), f"MSE: {mse:.4f}", fill=text_color, font=font)

        frames.append(np.array(img))

    # Hold final frame for 1.5 seconds.
    for _ in range(int(fps * 1.5)):
        frames.append(frames[-1])

    # Write MP4 (same codec/settings as visualize_trajectory.py).
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(output_path), frames, fps=fps,
                codec="libx264", macro_block_size=1)
