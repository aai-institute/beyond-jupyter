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

    def _get_tile(self, tx, ty):
        key = (tx, ty)
        tile = self._tiles.get(key)
        if tile is None:
            tile = MapTile(tx, ty, GeoRect(*self._mercator.TileLatLonBounds(tx, ty, self.zoom)), self.zoom)
            self._tiles[key] = tile
        return tile

    def iter_tile_coordinates_in_rect(self, rect: GeoRect) -> Generator[Tuple[int, int], None, None]:
        tx1, ty1 = self._mercator.LatLonToTile(rect.minLat, rect.minLon, self.zoom)
        tx2, ty2 = self._mercator.LatLonToTile(rect.maxLat, rect.maxLon, self.zoom)
        tx_min = min(tx1, tx2)
        tx_max = max(tx1, tx2)
        ty_min = min(ty1, ty2)
        ty_max = max(ty1, ty2)
        for tx in range(tx_min, tx_max+1):
            for ty in range(ty_min, ty_max+1):
                yield tx, ty

    def get_tiles_in_rect(self, rect: GeoRect) -> List[MapTile]:
        return [self._get_tile(tx, ty) for tx, ty in self.iter_tile_coordinates_in_rect(rect)]

    def get_tile(self, lat: float, lon: float) -> MapTile:
        return self._get_tile(*self.get_tile_coordinates(lat, lon))

    def get_tile_coordinates(self, lat: float, lon: float) -> Tuple[int, int]:
        return self._mercator.LatLonToTile(lat, lon, self.zoom)
