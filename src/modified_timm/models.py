from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TweakResult:
    name: str
    applied: bool
    reason: str
