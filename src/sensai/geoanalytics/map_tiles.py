"""
Utility functions and classes for geographic coordinates
"""

import math
from typing import Tuple, List, Generator

from ._globalmaptiles import GlobalMercator
from .geo_coords import GeoRect

EARTH_RADIUS = 6371000
EARTH_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS
LATITUDE_PER_METRE = 360.0 / EARTH_CIRCUMFERENCE



class MapTile:
    def __init__(self, tx: int, ty: int, rect: GeoRect, zoom: int):
        self.tx = tx
        self.ty = ty
        self.rect = rect
        self.zoom = zoom


class MapTiles:
    def __init__(self, zoom=13):
        self.zoom = zoom
        self._mercator = GlobalMercator()
        self._tiles = {}

    def _getTile(self, tx, ty):
        key = (tx, ty)
        tile = self._tiles.get(key)
        if tile is None:
            tile = MapTile(tx, ty, GeoRect(*self._mercator.TileLatLonBounds(tx, ty, self.zoom)), self.zoom)
            self._tiles[key] = tile
        return tile

    def iterTileCoordinatesInRect(self, rect: GeoRect) -> Generator[Tuple[int, int], None, None]:
        tx1, ty1 = self._mercator.LatLonToTile(rect.minLat, rect.minLon, self.zoom)
        tx2, ty2 = self._mercator.LatLonToTile(rect.maxLat, rect.maxLon, self.zoom)
        txMin = min(tx1, tx2)
        txMax = max(tx1, tx2)
        tyMin = min(ty1, ty2)
        tyMax = max(ty1, ty2)
        for tx in range(txMin, txMax+1):
            for ty in range(tyMin, tyMax+1):
                yield tx, ty

    def getTilesInRect(self, rect: GeoRect) -> List[MapTile]:
        return [self._getTile(tx, ty) for tx, ty in self.iterTileCoordinatesInRect(rect)]

    def getTile(self, lat: float, lon: float) -> MapTile:
        return self._getTile(*self.getTileCoordinates(lat, lon))

    def getTileCoordinates(self, lat: float, lon: float) -> Tuple[int, int]:
        return self._mercator.LatLonToTile(lat, lon, self.zoom)