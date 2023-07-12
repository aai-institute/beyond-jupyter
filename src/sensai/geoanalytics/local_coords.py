"""
Local coordinate systems (for geographic data)
"""
import math
from functools import reduce
from typing import Tuple, Union, List

import numpy as np
import utm
from shapely.geometry import polygon, multipolygon, point, LineString, mapping
from shapely.ops import polygonize, unary_union


class LocalCoordinateSystem(object):
    """
    Represents a local coordinate system for the conversion of geo-coordinates
    (latitude, longitude) to a local Cartesian coordinate system (unit=metre) and vice versa
    using the UTM transform
    """

    def __init__(self, lat, lon):
        """
        Parameters:
            lat: the latitude of the origin of the coordinate system
            lon: the longitude of the origin of the coordinate system
        """
        self.uRef = utm.from_latlon(lat, lon)
        self.uRefE = self.uRef[0]
        self.uRefN = self.uRef[1]
        self.uRefPseudoN = self._pseudoNorthing(self.uRefN)

    def getLocalCoords(self, lat, lon) -> Tuple[float, float]:
        uE, uN, zM, zL = utm.from_latlon(lat, lon)
        x = uE - self.uRefE
        y = self._pseudoNorthing(uN) - self.uRefPseudoN
        return x, y

    def getLatLon(self, localX, localY) -> Tuple[float, float]:
        easting = localX + self.uRefE
        pseudoNorthing = localY + self.uRefPseudoN
        return utm.to_latlon(easting, self._realNorthing(pseudoNorthing), self.uRef[2], self.uRef[3])

    @staticmethod
    def _pseudoNorthing(realNorthing):
        if realNorthing >= 10000000:
            return realNorthing - 10000000
        else:
            return realNorthing

    @staticmethod
    def _realNorthing(pseudoNorthing):
        if pseudoNorthing < 0:
            return pseudoNorthing + 10000000
        else:
            return pseudoNorthing


class LocalHexagonalGrid:
    """
    A local hexagonal grid, where hex cells can be referenced by two integer coordinates relative to
    the central grid cell, whose centre is at local coordinate (0, 0) and where positive x-coordinates/columns
    are towards the east and positive y-coordinates/rows are towards the north.
    Every odd row of cells is shifted half a hexagon to the right, i.e. column x for row 1 is half a grid cell
    further to the right than column x for row 0.

    For visualisation purposes, see https://www.redblobgames.com/grids/hexagons/
    """
    def __init__(self, radiusM):
        """
        :param radiusM: the radius, in metres, of each hex cell
        """
        self.radiusM = radiusM
        startAngle = math.pi / 6
        stepAngle = math.pi / 3
        self.offsetVectors = []
        for i in range(6):
            angle = startAngle + i * stepAngle
            x = math.cos(angle) * radiusM
            y = math.sin(angle) * radiusM
            self.offsetVectors.append(np.array([x, y]))
        self.hexagonWidth = 2 * self.offsetVectors[0][0]
        self.hexagonHeight = 2 * self.offsetVectors[1][1]
        self.rowStep = 0.75 * self.hexagonHeight
        self.polygonArea = 6 * self.hexagonHeight * self.hexagonWidth / 8

    def getHexagon(self, xColumn: int, yRow: int) -> polygon.Polygon:
        """
        Gets the hexagon (polygon) for the given integer hex cell coordinates
        :param xColumn: the column coordinate
        :param yRow: the row coordinate
        :return: the hexagon
        """
        centreX = xColumn * self.hexagonWidth
        centreY = yRow * self.rowStep
        if yRow % 2 == 1:
            centreX += 0.5 * self.hexagonWidth
        centre = np.array([centreX, centreY])
        return polygon.Polygon([centre + o for o in self.offsetVectors])

    def getMinHexagonColumn(self, x):
        lowestXDefinitelyInColumn0 = 0
        return math.floor((x - lowestXDefinitelyInColumn0) / self.hexagonWidth)

    def getMaxHexagonColumn(self, x):
        highestXDefinitelyInColumn0 = self.hexagonWidth / 2
        return math.ceil((x - highestXDefinitelyInColumn0) / self.hexagonWidth)

    def getMinHexagonRow(self, y):
        lowestYDefinitelyInRow0 = -self.hexagonHeight / 4
        return math.floor((y - lowestYDefinitelyInRow0) / self.rowStep)

    def getMaxHexagonRow(self, y):
        highestYDefinitelyInRow0 = self.hexagonHeight / 4
        return math.ceil((y - highestYDefinitelyInRow0) / self.rowStep)

    def getHexagonCoordSpanForBoundingBox(self, minX, minY, maxX, maxY) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Gets the range of hex-cell coordinates that cover the given bounding box

        :param minX: minimum x-coordinate of bounding box
        :param minY: minimum y-coordinate of bounding box
        :param maxX: maximum  x-coordinate of bounding box
        :param maxY: maximum y-coordinate of bounding box
        :return: a pair of pairs ((minCol, minRow), (maxCol, maxRow)) indicating the span of cell coordinates
        """
        if minX > maxX or minY > maxY:
            raise ValueError()
        minColumn = self.getMinHexagonColumn(minX)
        maxColumn = self.getMaxHexagonColumn(maxX)
        minRow = self.getMinHexagonRow(minY)
        maxRow = self.getMaxHexagonRow(maxY)
        return ((minColumn, minRow), (maxColumn, maxRow))

    def getHexagonCoordsForPoint(self, x, y):
        ((minColumn, minRow), (maxColumn, maxRow)) = self.getHexagonCoordSpanForBoundingBox(x, y, x, y)
        for xCol in range(minColumn, maxColumn+1):
            for yRow in range(minRow, maxRow+1):
                if self.getHexagon(xCol, yRow).contains(point.Point(x, y)):
                    return xCol, yRow
        raise Exception("No Hexagon matched; possible edge case (point on hexagon boundary)")


def fixPolygon(poly: Union[polygon.Polygon, multipolygon.MultiPolygon], maxAreaDiff=1e-2) -> Union[polygon.Polygon, multipolygon.MultiPolygon]:
    """
    Fix invalid shapely polygons or multipolygons.

    Reference:
    https://stackoverflow.com/questions/35110632/splitting-self-intersecting-polygon-only-returned-one-polygon-in-shapely

    :param poly: the polygon to fix
    :param maxAreaDiff: the maximum change in area
    :return: the fixed polygon or None if it cannot be fixed given the area change constraint
    """
    def _fixPolygonComponent(coords: List[Tuple[float, float]]):
        res = list(polygonize(unary_union(LineString(list(coords) + [coords[0]]))))
        return reduce(lambda p1, p2: p1.union(p2), res)

    if poly.is_valid:
        return poly
    else:
        if isinstance(poly, polygon.Polygon):
            exteriorCoords = poly.exterior.coords[:]
            fixedExterior = _fixPolygonComponent(exteriorCoords)
            fixedInterior = polygon.Polygon()
            for interior in poly.interiors:
                coords = interior.coords[:]
                fixedInterior = fixedInterior.union(_fixPolygonComponent(coords))
            fixedPolygon = fixedExterior.difference(fixedInterior)
        elif isinstance(poly, multipolygon.MultiPolygon):
            polys = list(poly)
            fixedPolys = [fixPolygon(p, maxAreaDiff=maxAreaDiff) for p in polys]
            fixedPolygon = reduce(lambda p1, p2: p1.union(p2), fixedPolys)
        else:
            raise Exception(f"Unsupported type {type(poly)}")
        areaDiff = float('Inf') if poly.area == 0 else abs(poly.area - fixedPolygon.area) / poly.area
        #log.info(f"Invalid polygon\n{poly}\nComputed fix:\n{fixedPolygon}.\nArea error: {areaDiff}")
        if areaDiff > maxAreaDiff:
            return None
        else:
            return fixedPolygon

