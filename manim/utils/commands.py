from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from subprocess import run
from typing import TypedDict

from manim.typing import StrOrBytesPath

__all__ = [
    "capture",
    "get_dir_layout",
]


def capture(
    command: str, cwd: StrOrBytesPath | None = None, command_input: str | None = None
) -> tuple[str, str, int]:
    p = run(
        command,
        cwd=cwd,
        input=command_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    out, err = p.stdout, p.stderr
    return out, err, p.returncode


class VideoMetadata(TypedDict):
    width: int
    height: int
    nb_frames: str
    duration: str
    avg_frame_rate: str
    codec_name: str
    pix_fmt: str



def get_dir_layout(dirpath: Path) -> Generator[str, None, None]:
    """Get list of paths relative to dirpath of all files in dir and subdirs recursively."""
    for p in dirpath.iterdir():
        if p.is_dir():
            yield from get_dir_layout(p)
            continue
        yield str(p.relative_to(dirpath))
