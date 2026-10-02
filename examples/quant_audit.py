"""
Look-ahead audit for the quant battery.

Every model is run twice: on the full history and on history cut off at
CUT. Any signal dated before the cut that differs between the two runs
was influenced by data the model could not have had -- a look-ahead leak,
however small. Models that pass produce identical signals up to the cut.

    python -m examples.quant_audit [keys...]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.quant.core import MODELS, Ctx                   # noqa: E402
from examples.quant import (models_daily, models_xasset,      # noqa: E402,F401
                            models_intraday, models_ml, models_risk,
                            models_extra)
from examples.quant.registry import entries, REFERENCES       # noqa: E402

CUT = "2014-07-01"


def compare(a, b, cut):
    """Largest absolute signal difference on rows strictly before the cut,
    leaving out the final bar before it (a bar whose close is the cut)."""
    idx = a.index
    cut_ts = pd.Timestamp(cut, tz=idx.tz) if idx.tz is not None else \
        pd.Timestamp(cut)
    rows = idx[idx < cut_ts]
    rows = rows[:-2] if len(rows) > 2 else rows
    a = a.reindex(rows).astype(float).fillna(0)
    b = b.reindex(index=rows, columns=a.columns).astype(float).fillna(0)
    d = (a - b).abs()
    return float(d.to_numpy().max()), float((d > 1e-9).to_numpy().mean())


def main():
    keys = sys.argv[1:] or list(dict.fromkeys(
        [k for _, k, _ in entries() if k] + REFERENCES))
    full, cut = Ctx(), Ctx(end=CUT)
    bad = []
    for k in keys:
        t = time.time()
        try:
            f = MODELS[k]["fn"]
            a, b = f(full), f(cut)
            mx, frac = compare(a, b, CUT)
            flag = "LEAK" if mx > 1e-6 else "ok"
            if flag == "LEAK":
                bad.append(k)
            print(f"{k:28s} {flag:4s} max diff {mx:.3g} on {frac:.2%} of "
                  f"cells  {time.time() - t:5.1f}s", flush=True)
        except Exception as e:
            print(f"{k:28s} ERROR {e!r}", flush=True)
    print("leaks:", bad)


if __name__ == "__main__":
    main()
