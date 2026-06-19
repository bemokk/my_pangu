import sys
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import certifi
except Exception:
    pass
else:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

try:
    import dask
except Exception:
    pass
else:
    _DASK_CONFIG = dask.config.set(scheduler="synchronous")
    try:
        import dask.base
    except Exception:
        pass
    else:
        dask.base._DISTRIBUTED_AVAILABLE = False

try:
    import xarray.backends.locks as xarray_locks
except Exception:
    pass
else:
    xarray_locks._get_scheduler = lambda *args, **kwargs: "threaded"
