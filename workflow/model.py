"""Bridge from the workflow to level6_cnn: re-exports FSRCNN and load_model."""

from methods.level6_cnn import FSRCNN, load_model
from methods.level6_cnn import _infer as infer

__all__ = ["FSRCNN", "load_model", "infer"]
