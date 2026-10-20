"""Project scripts.

Importing this package installs one compatibility shim (below) before anything
else in the pipeline loads.
"""


def _shim_scipy_propack():
    """Work around a scipy binary that macOS 27's dyld refuses to load.

    THE FAILURE
    -----------
    On macOS 27 (Darwin 27.0.0), loading scipy raises:

        dlopen(.../scipy/sparse/linalg/_propack/_spropack.cpython-310-darwin.so)
        section '__DATA/__thread_bss' has a zero-fill section type, but offset
        field is not zero

    The newer dyld rejects a malformed Mach-O section that older versions
    tolerated. It is in the PUBLISHED wheel, not a corrupt download -- a
    --force-reinstall --no-cache produces the identical binary.

    WHY IT BREAKS THIS PROJECT
    --------------------------
    Nothing here uses PROPACK. But `lerobot.policies.__init__` eagerly imports
    every policy, including `pi0_fast`, which imports `scipy.fftpack`, which
    cascades through scipy.linalg -> sparse -> _svdp -> _propack. So one unused
    Fortran binary takes down every live demo, while replay (which never imports
    lerobot) keeps working.

    WHY NOT JUST UPGRADE SCIPY
    --------------------------
    1.15.3 is the last release supporting Python 3.10; there is nothing newer to
    move to, and this environment is pinned to 3.10 by robosuite/lerobot.

    THE SHIM
    --------
    Register a stand-in module so the import chain completes. Attribute access
    returns a callable that raises only if actually invoked -- `_svdp.py` builds
    a dispatch table at import time (`_spropack.slansvd`), so attributes must
    resolve, but any real PROPACK call still fails loudly rather than silently
    returning wrong numbers.

    This is a no-op on machines where the binary loads normally.
    """
    import importlib
    import importlib.util
    import sys
    import types

    name = "scipy.sparse.linalg._propack"
    if name in sys.modules or importlib.util.find_spec("scipy") is None:
        return
    try:
        importlib.import_module(name)
        return                      # loads fine here; leave scipy alone
    except Exception:
        pass

    class _Unavailable:
        def __getattr__(self, attr):
            def _raise(*a, **k):
                raise RuntimeError(
                    f"scipy PROPACK ({attr}) cannot load on this macOS build. "
                    "Nothing in this project uses it; it is shimmed so lerobot "
                    "can import. See scripts/__init__.py:_shim_scipy_propack.")
            return _raise

    stub = types.ModuleType(name)
    for sym in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
        setattr(stub, sym, _Unavailable())
    sys.modules[name] = stub


_shim_scipy_propack()
