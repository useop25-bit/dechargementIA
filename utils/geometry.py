"""Shared dataclasses used as the data contract between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class BarcodeDetection:
    """
    One barcode decoded from the image.

    `value` is the FULL raw decoded payload exactly as the scanner returned
    it (e.g. "P9877021480"). `identifier` and `payload` are derived by
    splitting off the leading data-identifier letter (see
    config.settings.BARCODE_DATA_IDENTIFIERS) when the payload matches that
    convention — `identifier` is None for barcodes that don't match it (so
    nothing is silently discarded or misparsed).
    """

    value: str
    symbology: str
    confidence: float
    bbox: Optional[BBox] = None
    identifier: Optional[str] = None       # e.g. "P", "S" — None if unrecognized pattern
    identifier_meaning: Optional[str] = None  # e.g. "produit", "no_etiquette"
    payload: Optional[str] = None          # value with the identifier letter stripped
