"""SAGE public package."""

from importlib.metadata import version

from .foundation.config import SageConfig


__version__ = version("sage")
__all__ = ["SageConfig", "__version__"]
