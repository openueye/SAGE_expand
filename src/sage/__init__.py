"""SAGE public package."""

from importlib.metadata import version

from .config import SageConfig


__version__ = version("sage")
__all__ = ["SageConfig", "__version__"]
