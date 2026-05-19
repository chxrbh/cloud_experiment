"""Shared utilities for IIoT simulation experiment runners."""

from __future__ import annotations

import random
import sys
import time


def _readings(n: int, rng: random.Random) -> list[float]:
    return [round(rng.uniform(0, 150), 4) for _ in range(n)]


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(total, 1)
        self.label = label
        self.enabled = enabled
        self.current = 0
        self.started = time.perf_counter()
        self.last_message = ""

    def step(self, message: str = "") -> None:
        self.current += 1
        self.last_message = message
        self.render()

    def render(self, final: bool = False) -> None:
        if not self.enabled:
            return
        done = self.total if final else min(self.current, self.total)
        width = 32
        filled = int(width * done / self.total)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.perf_counter() - self.started
        suffix = f" | {self.last_message}" if self.last_message else ""
        sys.stdout.write(
            f"\r{self.label} [{bar}] {done}/{self.total}"
            f" {done / self.total * 100:5.1f}% {elapsed:6.1f}s{suffix}"
        )
        sys.stdout.flush()
        if final or done >= self.total:
            sys.stdout.write("\n")

    def finish(self) -> None:
        if self.current >= self.total:
            return
        self.current = self.total
        self.render(final=True)
