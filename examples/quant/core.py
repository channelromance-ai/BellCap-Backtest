"""
The model registry and the data context every model reads from.

A model is a function `f(X) -> frame` registered with `@model(key, ...)`.
It returns a signal frame (direction or weight per bar per symbol, known at
the bar's close). Options on the decorator tell the engine how to treat it:

  freq   "D" daily (17:00 New York roll), "H" hourly, "M" 15-minute
  scale  True: vol-scale each symbol to the same risk (default)
         False: the frame is already a portfolio notional
  agg    "mean" (default) or "sum" -- see bcbt.quant.run
  base   for overlays and sizing schemes: the key of the model they modify,
         so the results can report the change against it
  norm   sizing schemes: the net stream is rescaled to 10% realised vol
         before scoring, so schemes are compared at equal risk
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from bcbt import oanda_data, quant as Q                 # noqa: E402

MODELS: dict = {}


def model(key, freq="D", scale=True, agg="mean", base=None, norm=False):
    def deco(f):
        MODELS[key] = dict(fn=f, freq=freq, scale=scale, agg=agg, base=base,
                           norm=norm, doc=(f.__doc__ or "").strip())
        return f
    return deco


class Ctx:
    """Lazily built panels shared by every model."""

    def __init__(self, data_dir="data", end=None):
        self.data_dir = data_dir
        self.end = pd.Timestamp(end, tz="UTC") if end else None
        self._cache = {}

    def _cut(self, d):
        return d if self.end is None else d[d.index < self.end]

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @functools.cached_property
    def h1_long(self):
        return self._cut(oanda_data.load("1h"))

    @functools.cached_property
    def D(self):
        return Q.panels(self.h1_long, "D")

    @functools.cached_property
    def H(self):
        P = Q.panels(self.h1_long)
        # Only keep bars where at least a few markets trade (drops the
        # weekend stubs some feeds print).
        n = P["C"].notna().sum(axis=1)
        keep = n >= 5
        return {k: v[keep] for k, v in P.items()}

    @functools.cached_property
    def M(self):
        m = self._cut(oanda_data.load("15min"))
        P = Q.panels(m)
        n = P["C"].notna().sum(axis=1)
        return {k: v[n >= 5] for k, v in P.items()}

    @functools.cached_property
    def vix(self):
        v = pd.read_csv(os.path.join(self.data_dir, "vix_daily.csv"),
                        parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
        # VIX closes at 16:15 New York, inside the trading day that ends at
        # 17:00, so the same date lines up.
        return v.reindex(self.D["C"].index).ffill()

    @property
    def C(self):
        return self.D["C"]

    @functools.cached_property
    def R(self):
        return self.D["C"].pct_change(fill_method=None)

    @functools.cached_property
    def LR(self):
        return np.log(self.D["C"]).diff()


def hourly_rolls(index):
    """Number of 17:00 New York rolls between consecutive hourly bars."""
    day = Q.to_ny_day(index)
    d = pd.Series(day, index=index)
    nxt = d.shift(-1)
    return ((nxt - d).dt.days.fillna(0)).clip(lower=0)


def evaluate(key, X, sig=None, cost_mult=1.0):
    """Run one registered model through the engine."""
    m = MODELS[key]
    if sig is None:
        sig = m["fn"](X)
    P = {"D": X.D, "H": X.H, "M": X.M}[m["freq"]]
    C = P["C"]
    bpy = {"D": 252, "H": 252 * 23, "M": 252 * 23 * 4}[m["freq"]]
    rolls = None if m["freq"] == "D" else hourly_rolls(C.index)
    return Q.run(sig, C, bars_per_year=bpy, scale=m["scale"], rolls=rolls,
                 agg=m["agg"], cost_mult=cost_mult, name=key)
