"""
Image Processing: LetterboxPad
SRP — Resizes image while strictly preserving aspect ratio, padding with constant color to create a square canvas.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

from PIL import Image


class LetterboxPad:
    """
    Resizes image along its longest edge, then centers and pads with a fill color
    to generate a uniform square image of dimensions ``size x size``.

    Acts as the primary anatomical aspect-ratio preserving preprocessor.

    Parameters
    ----------
    size : int
        Dimension of the square output canvas.
    fill : tuple[int, int, int]
        RGB fill color for padding canvas. Defaults to black (0, 0, 0).
    """

    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"LetterboxPad(size={self.size}, fill={self.fill})"
