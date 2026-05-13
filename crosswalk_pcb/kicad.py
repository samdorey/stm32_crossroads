"""Parse KiCad .kicad_mod footprint files and extract SMD pad geometry.

Supports KiCad 6/7/8/9 .kicad_mod files (S-expression format).
We only need pad positions and sizes — the rendering and matching
happens in match.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_PAD_START = re.compile(r'\(pad\s+"([^"]+)"\s+(smd|thru_hole)\s+\w+')
_AT = re.compile(r'\(at\s+([\d.eE+-]+)\s+([\d.eE+-]+)(?:\s+([\d.eE+-]+))?')
_SIZE = re.compile(r'\(size\s+([\d.eE+-]+)\s+([\d.eE+-]+)')


@dataclass
class Pad:
    number: str
    x: float          # mm, from footprint origin
    y: float          # mm
    width: float       # mm (size X before rotation)
    height: float      # mm (size Y before rotation)
    angle: float       # degrees, from (at X Y ANGLE); 0 if absent
    mount: str         # 'smd' or 'thru_hole'


@dataclass
class Footprint:
    name: str
    path: str
    pads: list[Pad]

    @property
    def n_pads(self) -> int:
        return len(self.pads)


def parse_pads(path: Path) -> list[Pad]:
    """Extract all pads from a .kicad_mod file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pads: list[Pad] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = _PAD_START.search(lines[i])
        if not m:
            i += 1
            continue
        number, mount = m.group(1), m.group(2)
        x = y = w = h = angle = None
        for j in range(i, min(i + 15, len(lines))):
            if x is None:
                ma = _AT.search(lines[j])
                if ma:
                    x, y = float(ma.group(1)), float(ma.group(2))
                    angle = float(ma.group(3)) if ma.group(3) else 0.0
            if w is None:
                ms = _SIZE.search(lines[j])
                if ms:
                    w, h = float(ms.group(1)), float(ms.group(2))
            if x is not None and w is not None:
                break
        if x is not None and w is not None:
            pads.append(Pad(number=number, x=x, y=y, width=w, height=h,
                            angle=angle, mount=mount))
        i += 1
    return pads


def load_footprint(path: Path) -> Footprint | None:
    """Parse a single .kicad_mod. Returns None if <4 SMD pads."""
    pads = parse_pads(path)
    smd = [p for p in pads if p.mount == "smd"]
    if len(smd) < 4:
        return None
    return Footprint(name=path.stem, path=str(path), pads=smd)


def load_library(library_dir: Path, min_pads: int = 6) -> list[Footprint]:
    """Load all .kicad_mod files from a .pretty directory."""
    fps: list[Footprint] = []
    for f in sorted(library_dir.glob("*.kicad_mod")):
        fp = load_footprint(f)
        if fp is not None and fp.n_pads >= min_pads:
            fps.append(fp)
    return fps
