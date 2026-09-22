"""Concrete strategies. Each `run` returns a trade table with an `r` column."""

from . import ema_cross, orb_ema  # noqa: F401

__all__ = ["ema_cross", "orb_ema"]
