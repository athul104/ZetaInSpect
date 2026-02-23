# src/zetainspect/__init__.py
from importlib.metadata import PackageNotFoundError, version
from .background import InflationHistory
from .scales import Scales
from .spectrum import Spectrum

__all__ = ["InflationHistory", "Scales", "Spectrum", "__version__"]

try:
    __version__ = version("zetainspect")
except PackageNotFoundError:
    __version__ = "0+unknown"  # e.g., running from a source tree not installed