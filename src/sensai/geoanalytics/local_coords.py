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
        self.uRefPseudoN = self._pseudo_northing(self.uRefN)

    def get_local_coords(self, lat, lon) -> Tuple[float, float]:
        uE, uN, zM, zL = utm.from_latlon(lat, lon)
        x = uE - self.uRefE
        y = self._pseudo_northing(uN) - self.uRefPseudoN
        return x, y

    def get_lat_lon(self, local_x, local_y) -> Tuple[float, float]:
        easting = local_x + self.uRefE
        pseudo_northing = local_y + self.uRefPseudoN
        return utm.to_latlon(easting, self._real_northing(pseudo_northing), self.uRef[2], self.uRef[3])

    @staticmethod
    def _pseudo_northing(real_northing):
        if real_northing >= 10000000:
            return real_northing - 10000000
        else:
            return real_northing

    @staticmethod
    def _real_northing(pseudo_northing):
        if pseudo_northing < 0:
            return pseudo_northing + 10000000
        else:
            return pseudo_northing


class LocalHexagonalGrid:
    """
    A local hexagonal grid, where hex cells can be referenced by two integer coordinates relative to
    the central grid cell, whose centre is at local coordinate (0, 0) and where positive x-coordinates/columns
    are towards the east and positive y-coordinates/rows are towards the north.
    Every odd row of cells is shifted half a hexagon to the right, i.e. column x for row 1 is half a grid cell
    further to the right than column x for row 0.

    For visualisation purposes, see https://www.redblobgames.com/grids/hexagons/
    """
    def __init__(self, radius_m):
        """
        :param radius_m: the radius, in metres, of each hex cell
        """
        self.radius_m = radius_m
        start_angle = math.pi / 6
        step_angle = math.pi / 3
        self.offset_vectors = []
        for i in range(6):
            angle = start_angle + i * step_angle
            x = math.cos(angle) * radius_m
            y = math.sin(angle) * radius_m
            self.offset_vectors.append(np.array([x, y]))
        self.hexagon_width = 2 * self.offset_vectors[0][0]
        self.hexagon_height = 2 * self.offset_vectors[1][1]
        self.row_step = 0.75 * self.hexagon_height
        self.polygon_area = 6 * self.hexagon_height * self.hexagon_width / 8

    def get_hexagon(self, x_column: int, y_row: int) -> polygon.Polygon:
        """
        Gets the hexagon (polygon) for the given integer hex cell coordinates
        :param x_column: the column coordinate
        :param y_row: the row coordinate
        :return: the hexagon
        """
        centre_x = x_column * self.hexagon_width
        centre_y = y_row * self.row_step
        if y_row % 2 == 1:
            centre_x += 0.5 * self.hexagon_width
        centre = np.array([centre_x, centre_y])
        return polygon.Polygon([centre + o for o in self.offset_vectors])

    def get_min_hexagon_column(self, x):
        lowest_x_definitely_in_column0 = 0
        return math.floor((x - lowest_x_definitely_in_column0) / self.hexagon_width)

    def get_max_hexagon_column(self, x):
        highest_x_definitely_in_column0 = self.hexagon_width / 2
        return math.ceil((x - highest_x_definitely_in_column0) / self.hexagon_width)

    def get_min_hexagon_row(self, y):
        lowest_y_definitely_in_row0 = -self.hexagon_height / 4
        return math.floor((y - lowest_y_definitely_in_row0) / self.row_step)

    def get_max_hexagon_row(self, y):
        highest_y_definitely_in_row0 = self.hexagon_height / 4
        return math.ceil((y - highest_y_definitely_in_row0) / self.row_step)

    def get_hexagon_coord_span_for_bounding_box(self, min_x, min_y, max_x, max_y) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Gets the range of hex-cell coordinates that cover the given bounding box

        :param min_x: minimum x-coordinate of bounding box
        :param min_y: minimum y-coordinate of bounding box
        :param max_x: maximum  x-coordinate of bounding box
        :param max_y: maximum y-coordinate of bounding box
        :return: a pair of pairs ((minCol, min_row), (maxCol, max_row)) indicating the span of cell coordinates
        """
        if min_x > max_x or min_y > max_y:
            raise ValueError()
        min_column = self.get_min_hexagon_column(min_x)
        max_column = self.get_max_hexagon_column(max_x)
        min_row = self.get_min_hexagon_row(min_y)
        max_row = self.get_max_hexagon_row(max_y)
        return ((min_column, min_row), (max_column, max_row))

    def get_hexagon_coords_for_point(self, x, y):
        ((minColumn, minRow), (maxColumn, maxRow)) = self.get_hexagon_coord_span_for_bounding_box(x, y, x, y)
        for xCol in range(minColumn, maxColumn+1):
            for yRow in range(minRow, maxRow+1):
                if self.get_hexagon(xCol, yRow).contains(point.Point(x, y)):
                    return xCol, yRow
        raise Exception("No Hexagon matched; possible edge case (point on hexagon boundary)")


def fix_polygon(poly: Union[polygon.Polygon, multipolygon.MultiPolygon], maxAreaDiff=1e-2) \
        -> Union[polygon.Polygon, multipolygon.MultiPolygon]:
    """
    Fix invalid shapely polygons or multipolygons.

    Reference:
    https://stackoverflow.com/questions/35110632/splitting-self-intersecting-polygon-only-returned-one-polygon-in-shapely

    :param poly: the polygon to fix
    :param maxAreaDiff: the maximum change in area
    :return: the fixed polygon or None if it cannot be fixed given the area change constraint
    """
    def _fix_polygon_component(coords: List[Tuple[float, float]]):
        res = list(polygonize(unary_union(LineString(list(coords) + [coords[0]]))))
        return reduce(lambda p1, p2: p1.union(p2), res)

    if poly.is_valid:
        return poly
    else:
        if isinstance(poly, polygon.Polygon):
            exterior_coords = poly.exterior.coords[:]
            fixed_exterior = _fix_polygon_component(exterior_coords)
            fixed_interior = polygon.Polygon()
            for interior in poly.interiors:
                coords = interior.coords[:]
                fixed_interior = fixed_interior.union(_fix_polygon_component(coords))
            fixed_polygon = fixed_exterior.difference(fixed_interior)
        elif isinstance(poly, multipolygon.MultiPolygon):
            polys = list(poly)
            fixed_polys = [fix_polygon(p, maxAreaDiff=maxAreaDiff) for p in polys]
            fixed_polygon = reduce(lambda p1, p2: p1.union(p2), fixed_polys)
        else:
            raise Exception(f"Unsupported type {type(poly)}")
        area_diff = float('Inf') if poly.area == 0 else abs(poly.area - fixed_polygon.area) / poly.area
        #log.info(f"Invalid polygon\n{poly}\nComputed fix:\n{fixed_polygon}.\nArea error: {area_diff}")
        if area_diff > maxAreaDiff:
            return None
        else:
            return fixed_polygon

