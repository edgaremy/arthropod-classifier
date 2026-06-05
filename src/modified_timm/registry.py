from __future__ import annotations

from typing import Iterable, List, Protocol

from .models import TweakResult
from .tweaks import MacroF1ValidationTweak
from .tweaks import MacroF1ValidateScriptTweak


class TimmTweak(Protocol):
    name: str
    description: str

    def apply(self, train_module, strict: bool = True) -> TweakResult:
        ...


class TimmTweakRegistry:
    def __init__(self, tweaks: Iterable[TimmTweak]):
        self._tweaks: List[TimmTweak] = list(tweaks)

    def names(self) -> list[str]:
        return [t.name for t in self._tweaks]

    def describe(self) -> list[tuple[str, str]]:
        return [(t.name, t.description) for t in self._tweaks]

    def apply(self, train_module, disabled: set[str] | None = None, strict: bool = True) -> list[TweakResult]:
        disabled = disabled or set()
        results: list[TweakResult] = []
        for tweak in self._tweaks:
            if tweak.name in disabled:
                results.append(TweakResult(name=tweak.name, applied=False, reason='disabled by user'))
                continue
            results.append(tweak.apply(train_module=train_module, strict=strict))
        return results


def build_default_registry() -> TimmTweakRegistry:
    return TimmTweakRegistry([
        MacroF1ValidationTweak(),
    ])


def build_validation_registry() -> TimmTweakRegistry:
    return TimmTweakRegistry([
        MacroF1ValidateScriptTweak(),
    ])
