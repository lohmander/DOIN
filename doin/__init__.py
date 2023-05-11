"""Top-level package for DOIN."""
from doin.preprocess import preprocess
from doin.doin import build_model_from_clip

__all__ = ["preprocess", "build_model_from_clip"]
__author__ = """Hannes Lohmander"""
__email__ = "hannes@lohmander.org"
__version__ = "0.1.0"
