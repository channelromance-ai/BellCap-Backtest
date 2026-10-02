import sys, time, traceback
import pandas as pd
from examples.quant.core import MODELS, Ctx, evaluate
from bcbt import quant as Q
import examples.quant.models_daily  # noqa
import examples.quant.models_xasset  # noqa
import examples.quant.models_intraday  # noqa
import examples.quant.models_ml  # noqa
import examples.quant.models_risk  # noqa
import examples.quant.models_extra  # noqa
X = Ctx()
keys = sys.argv[1:] or list(MODELS)
for k in keys:
    t = time.time()
    try:
        res = evaluate(k, X)
        r = Q.score(res)
        print(f"{k:24s} S={r['sharpe']:6.2f} Sg={r['sharpe_gross']:6.2f} dev={r['sharpe_dev']:6.2f} hold={r['sharpe_hold']:6.2f} to={r['turnover']:.3f} exp={r['exposure']:.2f}  {time.time()-t:5.1f}s", flush=True)
    except Exception as e:
        print(k, "ERROR", repr(e)); traceback.print_exc()
