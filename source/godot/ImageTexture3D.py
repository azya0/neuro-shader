from typing import Optional

import numpy as np

from .ImageTextureParser import parse, Image


BASE_SIZE: int = 256

class GodotTexture3DSampler:
    def __init__(self, filename: str):
        self.images: list[Image] = parse(filename)
        self.depth = len(self.images)

        assert self.depth != 0

        self.width = self.images[0].width
        self.height = self.images[0].height
    
    def get(self, u: float, v: float, w: float) -> float:
        x = u * (self.width - 1)
        y = v * (self.height - 1)
        z = w * (self.depth - 1)
        
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        z0 = int(np.floor(z))
        
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)
        z1 = min(z0 + 1, self.depth - 1)
        
        wx = x - x0
        wy = y - y0
        wz = z - z0
        
        wx = np.clip(wx, 0, 1)
        wy = np.clip(wy, 0, 1)
        wz = np.clip(wz, 0, 1)
        
        v000 = self.images[z0].data[y0][x0]
        v001 = self.images[z0].data[y0][x1]
        v010 = self.images[z0].data[y1][x0]
        v011 = self.images[z0].data[y1][x1]
        v100 = self.images[z1].data[y0][x0]
        v101 = self.images[z1].data[y0][x1]
        v110 = self.images[z1].data[y1][x0]
        v111 = self.images[z1].data[y1][x1]
        
        c00 = self._lerp(v000, v001, wx)
        c01 = self._lerp(v010, v011, wx)
        c10 = self._lerp(v100, v101, wx)
        c11 = self._lerp(v110, v111, wx)
        
        c0 = self._lerp(c00, c01, wy)
        c1 = self._lerp(c10, c11, wy)
        
        result = self._lerp(c0, c1, wz)
        
        return result / 255.0
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        return a * (1 - t) + b * t
