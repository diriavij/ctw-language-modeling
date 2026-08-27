"""Context-tree models for sequential prediction.

Implementation follows van Veen (2007):
  "Using Context-Tree Weighting as a Language Modeler in Dasher"
  MSc thesis, TU Eindhoven.

Modules:
  binary_ctw  — BinaryCTW: integer-arithmetic binary CTW (Ch. 3–4)
  text_ctw    — TextCTW: practical interpolated approximation for text
  metrics     — bits_per_char(), perplexity() helpers
"""

from .binary_ctw import BinaryCTW
from .text_ctw import TextCTW
