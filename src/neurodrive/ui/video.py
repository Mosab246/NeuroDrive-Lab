"""Render a replay viewer to an MP4 file, off-screen, by piping frames to ffmpeg.

ffmpeg is taken from ``PATH``, or from the optional ``imageio-ffmpeg`` package
(``uv sync --extra video``) if that is installed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pygame

from neurodrive.ui.clock import SPEEDS
from neurodrive.ui.replay_viewer import ReplayViewer


class VideoError(RuntimeError):
    pass


def _show_speed(viewer: ReplayViewer, speed: float) -> None:
    """Make the panel's "playback" line match the video's speed (nearest listed speed)."""
    viewer.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError as exc:
        raise VideoError(
            "video export needs ffmpeg: install it and put it on PATH, or run "
            "`uv sync --extra video`"
        ) from exc


def export_video(
    viewer: ReplayViewer,
    path: str | Path,
    fps: int = 30,
    speed: float = 1.0,
    hold_end_s: float = 2.0,
    log: Callable[[str], None] = print,
) -> Path:
    """Play the viewer from its start to its end at ``speed`` and encode every frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = viewer.screen.get_size()
    cmd = [
        find_ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    _show_speed(viewer, speed)
    step = speed / fps
    t = viewer.t_start
    frames = 0
    try:
        while True:
            viewer.seek(t)
            viewer.draw()
            proc.stdin.write(pygame.image.tobytes(viewer.screen, "RGB"))
            frames += 1
            if t >= viewer.duration:
                break
            t = min(t + step, viewer.duration)
        last = pygame.image.tobytes(viewer.screen, "RGB")
        for _ in range(int(hold_end_s * fps)):
            proc.stdin.write(last)
            frames += 1
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise VideoError(f"ffmpeg failed with exit code {code}")
    log(f"wrote {path}: {frames} frames, {frames / fps:.1f} s at {fps} fps")
    return path


def export_timelapse(
    clips: list[tuple[ReplayViewer, str]],
    path: str | Path,
    fps: int = 30,
    speed: float = 2.0,
    hold_s: float = 1.0,
    log: Callable[[str], None] = print,
) -> Path:
    """Several replays one after another in one video, each with a caption banner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = clips[0][0].screen.get_size()
    cmd = [
        find_ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    frames = 0
    try:
        for viewer, caption in clips:
            viewer.caption = caption
            _show_speed(viewer, speed)
            t = viewer.t_start
            while True:
                viewer.seek(t)
                viewer.draw()
                proc.stdin.write(pygame.image.tobytes(viewer.screen, "RGB"))
                frames += 1
                if t >= viewer.duration:
                    break
                t = min(t + speed / fps, viewer.duration)
            last = pygame.image.tobytes(viewer.screen, "RGB")
            for _ in range(int(hold_s * fps)):
                proc.stdin.write(last)
                frames += 1
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise VideoError(f"ffmpeg failed with exit code {code}")
    log(f"wrote {path}: {len(clips)} clips, {frames / fps:.1f} s")
    return path
