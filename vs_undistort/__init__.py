import sys
from types import ModuleType
from .vs_undistort import vs_undistort

class _CallableModule(ModuleType):
    __call__ = staticmethod(vs_undistort)

sys.modules[__name__].__class__ = _CallableModule
__all__ = ["vs_undistort"]
